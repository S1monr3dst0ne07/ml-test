
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

    float* output_buffer;
    // prealloc'd output buffer
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

    size_t input_count  = layers_conf[0];
    size_t output_count = layers_conf[layer_count-1];

    //purely cosmetic: set input biases to zero
    for (size_t i = 0; i < input_count; i++)
        ls[0].ns[i].bias = 0;
        

    return (net_t) {
        .layers_conf = layers_conf,
        .count = layer_count,
        .ls = ls,
        .input_count  = input_count,
        .output_count = output_count,
        .output_buffer = malloc(sizeof(float) * output_count),
    };
}



float* nn_forward(net_t net, float* input)
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

    for (size_t i = 0; i < net.input_count; i++)
        net.output_buffer[i] = net.ls[net.count-1].ns[i].act;

    return net.output_buffer;
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

    size_t count;
    data_point_t* data;
} train_data_t;



float nn_loss(net_t n, train_data_t d)
{
    assert(n.input_count  == d.input_count);
    assert(n.output_count == d.output_count);

    float error = 0.0f;
    for (size_t i = 0; i < d.count; i++)
    {
        data_point_t p = d.data[i];
        float* prediction = nn_forward(n, p.inputs);

        for (size_t j = 0; j < d.output_count; j++)
        {
            float diff = prediction[j] - p.outputs[j];
            error += diff * diff;
        }
    }

    return error / d.count;
}



// computes backwards deltas for a single sample
void nn_backward(net_t net, data_point_t dp)
{
    // compute output layer deltas
    layer_t* output = &net.ls[net.count-1];
    for (size_t i = 0; i < output->count; i++)
    {
        neuron_t* n = &output->ns[i];
        float error = n->act - dp.outputs[i];
        n->delta = error * NN_ACT_DERI(n->act);
    }

    for (size_t i = net.count-2; i > 0; i--)
    {
        layer_t* curr_layer = &net.ls[i];
        layer_t* next_layer = &net.ls[i+1];

        for (size_t j = 0; j < curr_layer->count; j++)
        {
            neuron_t* curr = &curr_layer->ns[j];
            float downstream = 0.0f;

            for (size_t k = 0; k < next_layer->count; k++)
            {
                neuron_t* next = &next_layer->ns[k];
                downstream += next->weights[j] * next->delta;
            }

            curr->delta = downstream * NN_ACT_DERI(curr->act);
        }
    }
}


void nn_gradient(net_t net, float rate)
{
    for (size_t i = 1; i < net.count; i++)
    {
        layer_t* prev_layer = &net.ls[i-1];
        layer_t* curr_layer = &net.ls[i];

        for (size_t j = 0; j < curr_layer->count; j++)
        {
            neuron_t* n = &curr_layer->ns[j];

            for (size_t w = 0; w < n->weights_count; w++)
                n->weights[w] -= rate * n->delta * prev_layer->ns[w].act;

            n->bias -= rate * n->delta;
        }
    }
}


float nn_train(net_t net, train_data_t train, float rate)
{
    float total_loss = 0.0f;

    for (size_t i = 0; i < train.count; i++)
    {
        data_point_t dp = train.data[i];
        float* output = nn_forward(net, dp.inputs);

        //compute loss MSE
        for (size_t j = 0; j < train.output_count; j++)
        {
            float diff = output[j] - dp.outputs[j];
            total_loss += diff * diff;
        }

        nn_backward(net, dp);
        nn_gradient(net, rate);
    }

    return total_loss / train.count;
}
