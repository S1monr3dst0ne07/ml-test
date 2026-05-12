
#include <time.h>
#include <stdio.h>

#include "nn.c"


train_data_t d = {
    .data = (data_point_t[]){
        { .inputs = (float[]){ 0.0f, 0.0f }, .outputs = (float[]){ 0.0f } },
        { .inputs = (float[]){ 1.0f, 0.0f }, .outputs = (float[]){ 1.0f } },
        { .inputs = (float[]){ 0.0f, 1.0f }, .outputs = (float[]){ 1.0f } },
        { .inputs = (float[]){ 1.0f, 1.0f }, .outputs = (float[]){ 0.0f } },
    },
    .count = 4,
    .scratch_output = (float[4]){ 0 },
    .input_count = 2,
    .output_count = 1,
};



const float eps = 1e-3;
const float rate = 1;

int main()
{

    srand(time(0));

    net_t n = nn_create_net((size_t[]){2, 2, 1, 0});

    for (int i = 0; i < 100000; i++)
    {
        net_t grad = nn_yiff(n, d, eps);
        nn_learn(n, grad, rate);
    }
    float c = nn_cost(n, d);
    printf("cost: %f\n", c);


    printf("----------- full net -----------\n");
    for (size_t i = 0; i < d.count; i++)
    {
        data_point_t dp = d.data[i];
        nn_forward(n, dp.inputs, d.scratch_output);
        printf("%f | %f = %f\n", dp.inputs[0], dp.inputs[1], d.scratch_output[0]);
    }

    return 0; 
}



