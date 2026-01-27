// Reference single-threaded training loop.
//
// This file trains a character-level language model with an MLP backbone. It serves 
// as a reference implementation of end-to-end training and inference with no 
// special parallelisms applied.
//
// To run:
//     mpicc -O3 parallelisms/train.c && mpirun -n 1 a.out
//
// Inspired by Bengio et. al [1] and Karpath's makemore [2].
//   [1] https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
//   [2] https://github.com/karpathy/makemore 


#include <stdlib.h>
#include <string.h>
#include "src/data.c"
#include "src/model.c"


int main() {
    int batch_size = 32;
    int seq_len = 16;  // seq_len is computed offline and is equal to the longest word.
    int vocab_size = 27; 
    int emb_size = 16;
    int hidden_size = 4 * emb_size;

    srand(42);

    // Create dataset.
    Dataset* dataset = Dataset_create_from_file("data/names.txt", seq_len);
    Dataset train_split, test_split;
    Dataset_train_test_split(dataset, &train_split, &test_split, /* train_percent */ 0.9);
    int* Xs = malloc(sizeof(int) * batch_size * seq_len);
    int* Ys = malloc(sizeof(int) * batch_size);

    // Train.
    float lr = 0.1;
    int steps = 25000;
    Model* model = Model_create(batch_size, seq_len, vocab_size, emb_size, hidden_size);
    for (int step = 0; step < steps; step++) {
        Dataset_get_batch(&train_split, Xs, Ys, batch_size);
        float loss = Model_forward(model, Xs, Ys);
        printf("step: %d, loss %f\n", step, loss);
        Model_backward(model, Xs, Ys);
        Model_step(model, lr);
    }

    // Validate.
    float loss_acc = 0.0f;
    int n_valid_batches = 100;
    for (int i = 0; i < n_valid_batches; i ++) {
        Dataset_get_batch(&test_split, Xs, Ys, batch_size);
        loss_acc += Model_forward(model, Xs, Ys);
    }
    printf("Final validation loss: %f\n", loss_acc / n_valid_batches);

    // Sample.
    int sample_batch_size = 1;
    int* sample_Xs = calloc(sizeof(float), batch_size * seq_len);
    int* dummy_Ys = calloc(sizeof(float), batch_size);
    for (int i = 0; i < 10; i++)  {
        Model_sample(model, sample_Xs, dummy_Ys, seq_len);
        Dataset_print_batch(sample_Xs, dummy_Ys, sample_batch_size, seq_len);
        memset(sample_Xs, 0, sizeof(float) * batch_size * seq_len);
   }

   return 0;
}
