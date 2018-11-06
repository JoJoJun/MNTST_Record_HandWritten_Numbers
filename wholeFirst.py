import numpy
# this is for sigmoid function
import scipy.special
from matplotlib import pyplot as plt

# plt.imshow(random_image, cmap='gray')
# plt.colorbar()
# plt.show()

class neuralNetwork:
    def __init__(self,input_nodes,hidden_nodes, output_nodes,learning_rate):
        # set the number of notes in each layers & lr & link_weight & ativation_function
        self.inpput_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        self.wih = numpy.random.normal(0.0, pow(self.hidden_nodes,-0.5),(self.hidden_nodes,self.inpput_nodes))

        self.who = numpy.random.normal(0.0,pow(self.output_nodes,-0.5),(self.output_nodes,self.hidden_nodes))

        self.activation_func = lambda x:scipy.special.expit(x)
        pass

    # train the neural network
    def train(self,inputs_list, target_list):
        # convert the inputlist & targetlist into 2D array
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(target_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        output_errors = targets - final_outputs
        # hidden layer error is the output layer errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T,output_errors)

        # update the weights for the links
        self.who += self.learning_rate*numpy.dot(output_errors*final_outputs*(1.0-final_outputs),numpy.transpose(hidden_outputs))

        self.wih += self.learning_rate*numpy.dot(hidden_errors*hidden_outputs*(1.0-hidden_errors),numpy.transpose(inputs))
        pass

    def query(self,inputs_list):
        inputs = numpy.array(inputs_list,ndmin=2).T

        hidden_inputs = numpy.dot(self.wih,inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.who,hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

if __name__ == '__main__':
    # the numbers of nodes, the MNIST's picture is 28*28pix
        input_nodes = 784
        hidden_nodes = 200
        output_node = 10

        learning_rate = 0.1

        n = neuralNetwork(input_nodes,hidden_nodes,output_node,learning_rate)
    # read the training data
        training_datafile = ''
        training_data_list = []
        with open(training_datafile,'r')as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                training_data_list.append(line)

    # this is the number of times the training data set is used for training
        epoches = 5

        for e in range(epoches):
            for record in training_data_list:
                all_values = record.split(',')
                # scale and shift the inputs
                inputs = (numpy.asfarray(all_values[1:])/255.0 * 0.99)+0.01
                # create the target outouts
                targets = numpy.zeros(output_node)+0.01
                targets[int(all_values[0])] = 0.99
                n.train(inputs,targets)
                pass
            pass

    # load the test data
        test_datafile = ''
        test_data_list = []
        with open(test_datafile,'r')as rf:
            while True:
                line = rf.readline()
                if not line:
                    break
                test_data_list.append(line)
    # score for how well the network performs
        scorecard = []
        for record in test_data_list:
            all_values = record.split(',')
            correct_label = int(all_values[0])
            inputs = numpy.asfarray(all_values[1:])/255.0*0.99 + 0.01
            outputs = n.query(inputs)
            # the index of the highest value corresponds to list
            label = numpy.argmax(outputs)

            if(label == correct_label):
                scorecard.append(1)
            else:
                scorecard.append(0)
                pass
            pass


        scorecard_array = numpy.asarray(scorecard)
        print("performance = ",scorecard_array.sum()/scorecard_array.size)




