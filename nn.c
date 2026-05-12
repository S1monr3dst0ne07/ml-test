
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>



float nn_rand_float()
{
    return (float)rand() / RAND_MAX;
}
float nn_sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}
float nn_dsigmoidf(float y)
{
    return y * (1 - y);
}

#define NN_ACT_FUNC nn_sigmoidf
#define NN_ACT_DERI nn_dsigmoidf



typedef struct
{
    float act;
    float delta;

    float  bias;
    size_t weights_count;
    float* weights;
} neuron_t;


neuron_t nn_create_neuron(size_t conn)
{
    float* weight = malloc(sizeof(float) * conn);

    for (size_t i = 0; i < conn; i++)
        weight[i] = nn_rand_float();

    return (neuron_t) {
        .bias = nn_rand_float(),
        .weights_count = conn,
        .weights = weight
    };
}


typedef struct
{
    size_t count;
    neuron_t* ns;
} layer_t;


layer_t nn_create_layer(size_t prev, size_t this)
{
    neuron_t* ns = malloc(sizeof(neuron_t) * this);

    for (size_t i = 0; i < this; i++)
        ns[i] = nn_create_neuron(prev);

    return (layer_t) {
        .count = this,
        .ns = ns,
    };
}


typedef struct 
{
    size_t* layers_conf; 
    // -external, don't dealloc
    // -null terminated

    size_t count;
    layer_t* ls;

    size_t input_count;
    size_t output_count;
} net_t;



net_t nn_create_net(size_t layers_conf[])
{
    size_t layer_count = 0;
    while (layers_conf[layer_count]) layer_count++;

    layer_t * ls = malloc(sizeof(layer_t) * layer_count);

    for (size_t i = 0; i < layer_count; i++)
        ls[i] = nn_create_layer(
            i > 0 ? layers_conf[i-1] : 0, //input neurons don't have weights
            layers_conf[i]
        );

    return (net_t) {
        .layers_conf = layers_conf,
        .count = layer_count,
        .ls = ls,
        .input_count  = layers_conf[0],
        .output_count = layers_conf[layer_count-1],
    };
}



void nn_forward(net_t net, float* input, float* output)
{
    for (size_t i = 0; i < net.input_count; i++)
        net.ls[0].ns[i].act = input[i];

    for (size_t layer_i = 1; layer_i < net.count; layer_i++)
        //skip input layer
    {
        layer_t* prev_layer = &net.ls[layer_i-1];
        layer_t* curr_layer = &net.ls[layer_i];

        for (size_t neu_i = 0; neu_i < curr_layer->count; neu_i++)
        {
            neuron_t* curr_neuron = &curr_layer->ns[neu_i];
            float sum = curr_neuron->bias;

            assert(prev_layer->count == curr_neuron->weights_count);

            for (size_t conn_i = 0; conn_i < prev_layer->count; conn_i++)
            {
                neuron_t* prev_neuron = &prev_layer->ns[conn_i];
                sum += curr_neuron->weights[conn_i] * prev_neuron->act;
            }

            curr_neuron->act = NN_ACT_FUNC(sum); 
        }
    }

    if (output == NULL) return;

    for (size_t i = 0; i < net.input_count; i++)
        output[i] = net.ls[net.count-1].ns[i].act;
}







typedef struct
{
    float* inputs;
    float* outputs;
} data_point_t;


typedef struct
{
    size_t input_count;
    size_t output_count;

    float* scratch_output;

    size_t count;
    data_point_t* data;
} train_data_t;



float nn_cost(net_t n, train_data_t d)
{
    assert(n.input_count  == d.input_count);
    assert(n.output_count == d.output_count);

    float error = 0.0f;
    for (size_t i = 0; i < d.count; i++)
    {
        data_point_t p = d.data[i];
        nn_forward(n, p.inputs, d.scratch_output);

        for (size_t j = 0; j < d.output_count; j++)
        {
            float delta = d.scratch_output[j] - p.outputs[j];
            error += delta * delta;
        }
    }

    return error / d.count;
}


net_t nn_yiff(net_t net, train_data_t d, float eps)
{
    float c = nn_cost(net, d);
    net_t grad = nn_create_net(net.layers_conf);

    for (size_t layer_i = 1; layer_i < net.count; layer_i++)
    {
        layer_t net_layer  = net.ls[layer_i];
        layer_t grad_layer = grad.ls[layer_i];

        for (size_t neu_i = 0; neu_i < net_layer.count; neu_i++)
        {
            neuron_t* net_neuron  = &net_layer.ns[neu_i];
            neuron_t* grad_neuron = &grad_layer.ns[neu_i];

            for (size_t weight_i = 0; weight_i < net_neuron->weights_count; weight_i++)
            {
                net_neuron->weights[weight_i] += eps;
                grad_neuron->weights[weight_i] = (nn_cost(net, d) - c) / eps;
                net_neuron->weights[weight_i] -= eps;
            }

            net_neuron->bias += eps;
            grad_neuron->bias = (nn_cost(net, d) - c) / eps;
            net_neuron->bias -= eps;
        }
    }

    return grad;
}





void nn_learn(net_t net, net_t grad, float rate)
{
    for (size_t layer_i = 0; layer_i < net.count; layer_i++)
    {
        layer_t net_layer  = net.ls[layer_i];
        layer_t grad_layer = grad.ls[layer_i];

        for (size_t neu_i = 0; neu_i < net_layer.count; neu_i++)
        {
            neuron_t* net_neuron  = &net_layer.ns[neu_i];
            neuron_t* grad_neuron = &grad_layer.ns[neu_i];

            for (size_t weight_i = 0; weight_i < net_neuron->weights_count; weight_i++)
                net_neuron->weights[weight_i] -= grad_neuron->weights[weight_i] * rate;

            net_neuron->bias -= grad_neuron->bias * rate;
        }
    }
}




