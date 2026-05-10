
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

#define NN_ACT_FUNC nn_sigmoidf



typedef struct
{
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
    size_t  layers_conf_count;

    size_t count;
    layer_t* ls;

    size_t input_count;
    size_t output_count;

    //activation buffers 
    size_t width;
    float* src;
    float* dst;
} net_t;



net_t nn_create_net(size_t layers_conf[], size_t layer_count)
{
    layer_t * ls = malloc(sizeof(layer_t) * layer_count);

    for (size_t i = 0; i < layer_count-1; i++)
        ls[i] = nn_create_layer(layers_conf[i], layers_conf[i+1]);

    size_t width = 0;
    for (size_t i = 0; i < layer_count; i++)
        if (width < layers_conf[i]) width = layers_conf[i];

    float* b1 = malloc(sizeof(float) * width);
    float* b2 = malloc(sizeof(float) * width);

    size_t* layer_conf_cpy = malloc(sizeof(size_t) * layer_count);
    memcpy(layer_conf_cpy, layers_conf, sizeof(size_t) * layer_count);

    return (net_t) {
        .layers_conf = layer_conf_cpy,
        .layers_conf_count = layer_count,
        .count = layer_count-1,
        .ls = ls,
        .input_count  = layers_conf[0],
        .output_count = layers_conf[layer_count-1],
        .width = width,
        .src = b1,
        .dst = b2,
    };
}



static inline void nn_fcpy(float* dst, float* src, size_t n)
{
    memcpy(dst, src, n * sizeof(float));
}



void nn_forward(net_t net, float* input, float* output)
{
    nn_fcpy(net.src, input, net.input_count);

    for (size_t layer_i = 0; layer_i < net.count; layer_i++)
    {
        layer_t layer = net.ls[layer_i];

        for (size_t neu_i = 0; neu_i < layer.count; neu_i++)
        {
            neuron_t neuron = layer.ns[neu_i];
            float sum = neuron.bias;

            for (size_t conn_i = 0; conn_i < neuron.weights_count; conn_i++)
            {
                sum += neuron.weights[conn_i] * net.src[conn_i];
            }

            net.dst[neu_i] = NN_ACT_FUNC(sum);
        }

        // pointer swap
        float* src_ptr = net.src;
        float* dst_ptr = net.dst;

        net.src = dst_ptr;
        net.dst = src_ptr;
    }

    // read from source because it's been pointer swapped.
    // never used net.dst!!!
    nn_fcpy(output, net.src, net.output_count);
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
    net_t grad = nn_create_net(net.layers_conf, net.layers_conf_count);

    for (size_t layer_i = 0; layer_i < net.count; layer_i++)
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




