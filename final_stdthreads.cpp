#ifdef _MSC_VER
#define NOMINMAX
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <array>
#include <string>
#include <fstream>
#include <algorithm>
#include <thread>

typedef unsigned char byte;

#define SIGNATURE_LEN 64

int DENSITY = 21;
int PARTITION_SIZE;

int inverse[256];
const char* alphabet = "CSTPAGNDEQHRKMILVFYW";
const int ALPHABET_SIZE = 20;

void seed_random(char* term, int length);
short random_num(short max);

int WORDLEN;
FILE* sig_file;

std::vector<short> vocab_signatures;
using SignatureBytes = std::array<byte, SIGNATURE_LEN / 8>;

short* compute_new_term_sig(char* term, short* term_sig)
{
    seed_random(term, WORDLEN);

    int non_zero = SIGNATURE_LEN * DENSITY / 100;

    int positive = 0;
    while (positive < non_zero / 2)
    {
        short pos = random_num(SIGNATURE_LEN);
        if (term_sig[pos] == 0)
        {
            term_sig[pos] = 1;
            positive++;
        }
    }

    int negative = 0;
    while (negative < non_zero / 2)
    {
        short pos = random_num(SIGNATURE_LEN);
        if (term_sig[pos] == 0)
        {
            term_sig[pos] = -1;
            negative++;
        }
    }
    return term_sig;
}

static inline int term_to_index(const char* term)
{
    int index = 0;
    for (int i = 0; i < WORDLEN; ++i)
    {
        unsigned char c = (unsigned char)term[i];
        int value = inverse[c];
        if (value < 0 || value >= ALPHABET_SIZE) return -1;
        index = index * ALPHABET_SIZE + value;
    }
    return index;
}

short* find_sig(const char* term)
{
    int index = term_to_index(term);
    if (index < 0) return NULL;
    size_t offset = (size_t)index * SIGNATURE_LEN;
    if (offset + SIGNATURE_LEN > vocab_signatures.size()) return NULL;
    return &vocab_signatures[offset];
}

SignatureBytes compute_signature_chunk(const char* sequence, int length)
{
    SignatureBytes packed{};
    if (length < WORDLEN) return packed;

    const int windows = length - WORDLEN + 1;
    int doc_sig_local[SIGNATURE_LEN] = { 0 };

    for (int i = 0; i < windows; ++i)
    {
        short* term_sig = find_sig(sequence + i);
        if (!term_sig) continue;
        for (int j = 0; j < SIGNATURE_LEN; ++j)
            doc_sig_local[j] += term_sig[j];
    }

    for (int i = 0; i < SIGNATURE_LEN; i += 8)
    {
        byte c = 0;
        for (int j = 0; j < 8; j++)
            c |= (doc_sig_local[i + j] > 0) << (7 - j);
        packed[i / 8] = c;
    }
    return packed;
}

std::vector<SignatureBytes> partition_sequence(const char* sequence, int length)
{
    std::vector<SignatureBytes> signatures;
    if (length <= 0) return signatures;

    int step = PARTITION_SIZE / 2;
    if (step <= 0) step = 1;

    std::vector<int> offsets;
    for (int i = 0;;)
    {
        offsets.push_back(i);
        if (i + step >= length) break;
        i += step;
    }

    signatures.reserve(offsets.size());
    for (size_t idx = 0; idx < offsets.size(); ++idx)
    {
        int start = offsets[idx];
        int chunk_len = std::min(PARTITION_SIZE, length - start);
        signatures.push_back(compute_signature_chunk(sequence + start, chunk_len));
    }

    return signatures;
}

int power(int n, int e)
{
    int p = 1;
    for (int j = 0; j < e; j++)
        p *= n;
    return p;
}

void process_range(const std::vector<std::string>& sequences,
    std::vector<std::vector<SignatureBytes>>& doc_signatures,
    int start_index,
    int end_index)
{
    for (int doc = start_index; doc < end_index; ++doc)
    {
        const std::string& seq = sequences[doc];
        doc_signatures[doc] = partition_sequence(seq.c_str(), (int)seq.size());
    }
}

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        printf("Usage: %s <num_threads> [input_fasta]\n", argv[0]);
        return 1;
    }

    int threads = atoi(argv[1]);
    if (threads <= 0)
    {
        fprintf(stderr, "Error: num_threads must be > 0 (got %d)\n", threads);
        return 1;
    }

    const char* filename = "qut3.fasta";
    if (argc > 2) filename = argv[2];

    WORDLEN = 3;
    PARTITION_SIZE = 2048;
    int WORDS = power(ALPHABET_SIZE, WORDLEN);

    memset(inverse, -1, sizeof(inverse));
    for (int i = 0; i < (int)strlen(alphabet); i++)
        inverse[(unsigned char)alphabet[i]] = i;

    vocab_signatures.assign((size_t)WORDS * SIGNATURE_LEN, 0);
    std::vector<char> term(WORDLEN + 1, 0);
    term[WORDLEN] = '\0';
    for (int idx = 0; idx < WORDS; ++idx)
    {
        int value = idx;
        for (int pos = WORDLEN - 1; pos >= 0; --pos)
        {
            int digit = value % ALPHABET_SIZE;
            term[pos] = alphabet[digit];
            value /= ALPHABET_SIZE;
        }
        short* sig = &vocab_signatures[(size_t)idx * SIGNATURE_LEN];
        memset(sig, 0, SIGNATURE_LEN * sizeof(short));
        compute_new_term_sig(term.data(), sig);
    }

    std::vector<std::string> sequences;
    std::ifstream file(filename);
    if (!file.is_open())
    {
        fprintf(stderr, "Error: failed to open file %s\n", filename);
        return 1;
    }

    std::string line;
    std::string current_sequence;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();

        if (!line.empty() && line[0] == '>')
        {
            if (!current_sequence.empty())
            {
                sequences.push_back(std::move(current_sequence));
                current_sequence.clear();
            }
        }
        else if (!line.empty())
        {
            current_sequence += line;
        }
    }
    if (!current_sequence.empty())
        sequences.push_back(std::move(current_sequence));

    std::vector<std::vector<SignatureBytes>> doc_signatures(sequences.size());

    unsigned int hw_threads = std::thread::hardware_concurrency();
    if (hw_threads == 0) hw_threads = 1;

    if (threads > (int)hw_threads)
        threads = (int)hw_threads;
    if (threads < 1)
        threads = 1;

    int total_docs = (int)sequences.size();
    if (total_docs == 0)
        threads = 0;
    else if (threads > total_docs)
        threads = total_docs;

    printf("Using %d thread%s\n", threads, threads == 1 ? "" : "s");

    auto t_begin = std::chrono::high_resolution_clock::now();

    if (threads > 0)
    {
        std::vector<std::thread> workers;
        workers.reserve((size_t)threads);

        int base = total_docs / threads;
        int remainder = total_docs % threads;
        int start_index = 0;

        for (int t = 0; t < threads; ++t)
        {
            int count = base + (t < remainder ? 1 : 0);
            int end_index = start_index + count;
            workers.emplace_back(process_range,
                std::cref(sequences),
                std::ref(doc_signatures),
                start_index,
                end_index);
            start_index = end_index;
        }

        for (auto& worker : workers)
            worker.join();
    }

    auto t_compute = std::chrono::high_resolution_clock::now();

    char outfile[256];
    snprintf(outfile, sizeof(outfile), "%s.part%d_sigs%02d_%d", filename, PARTITION_SIZE, WORDLEN, SIGNATURE_LEN);

#ifdef _MSC_VER
    if (fopen_s(&sig_file, outfile, "wb") != 0)
    {
        fprintf(stderr, "Error: failed to open output file %s\n", outfile);
        return 1;
    }
#else
    sig_file = fopen(outfile, "wb");
    if (sig_file == NULL)
    {
        fprintf(stderr, "Error: failed to open output file %s\n", outfile);
        return 1;
    }
#endif

    for (int doc = 0; doc < (int)doc_signatures.size(); ++doc)
    {
        const auto& partitions = doc_signatures[doc];
        for (const auto& sig_bytes : partitions)
        {
            fwrite(&doc, sizeof(int), 1, sig_file);
            fwrite(sig_bytes.data(), sizeof(byte), sig_bytes.size(), sig_file);
        }
    }
    fclose(sig_file);

    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> compute_time = t_compute - t_begin;
    std::chrono::duration<double> total_time = t_end - t_begin;

    printf("%s compute %f seconds\n", filename, compute_time.count());
    printf("%s total   %f seconds\n", filename, total_time.count());

    return 0;
}
