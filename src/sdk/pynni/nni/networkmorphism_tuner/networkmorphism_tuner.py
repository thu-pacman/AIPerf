# Copyright (c) Microsoft Corporation. 
# Copyright (c) Tsinghua University.
# Copyright (c) Peng Cheng Laboratory.
# All rights reserved.
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
# OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ==================================================================================================

import logging
import os

import logging


from nni.tuner import Tuner
from nni.utils import OptimizeMode, extract_scalar_reward
from nni.networkmorphism_tuner.bayesian import BayesianOptimizer
from nni.networkmorphism_tuner.nn import CnnGenerator, MlpGenerator
from nni.networkmorphism_tuner.utils import Constant

from nni.networkmorphism_tuner.graph import graph_to_json, json_to_graph
import time
import nni
import multiprocessing

logger = logging.getLogger("NetworkMorphism_AutoML")
lock=multiprocessing.Lock()

class NetworkMorphismTuner(Tuner):
    """NetworkMorphismTuner is a tuner which using network morphism techniques."""

    def __init__(
            self,
            task="cv",
            input_width=32,
            input_channel=3,
            n_output_node=10,
            algorithm_name="Bayesian",
            optimize_mode="maximize",
            path="model_path",
            verbose=True,
            beta=Constant.BETA,
            t_min=Constant.T_MIN,
            max_model_size=Constant.MAX_MODEL_SIZE,
            default_model_len=Constant.MODEL_LEN,
            default_model_width=Constant.MODEL_WIDTH,
            init_model_dir= Constant.FILE_DIR,
    ):
        """ initilizer of the NetworkMorphismTuner.

        Parameters
        ----------
        task : str
            task mode, such as "cv","common" etc. (default: {"cv"})
        input_width : int
            input sample shape (default: {32})
        input_channel : int
            input sample shape (default: {3})
        n_output_node : int
            output node number (default: {10})
        algorithm_name : str
            algorithm name used in the network morphism (default: {"Bayesian"})
        optimize_mode : str
            optimize mode "minimize" or "maximize" (default: {"minimize"})
        path : str
            default mode path to save the model file (default: {"model_path"})
        verbose : bool
            verbose to print the log (default: {True})
        beta : float
            The beta in acquisition function. (default: {Constant.BETA})
        t_min : float
            The minimum temperature for simulated annealing. (default: {Constant.T_MIN})
        max_model_size : int
            max model size to the graph (default: {Constant.MAX_MODEL_SIZE})
        default_model_len : int
            default model length (default: {Constant.MODEL_LEN})
        default_model_width : int
            default model width (default: {Constant.MODEL_WIDTH})
        init_model_list : list
            store the remorph models , every node initializes
        """

        if not os.path.exists(path):
            os.makedirs(path)
        self.path = os.path.join(os.getcwd(), path)
        if task == "cv":
            self.generators = [CnnGenerator]
        elif task == "common":
            self.generators = [MlpGenerator]
        else:
            raise NotImplementedError('{} task not supported in List ["cv","common"]')

        self.n_classes = n_output_node
        self.input_shape = (input_width, input_width, input_channel)

        self.t_min = t_min
        self.beta = beta
        self.algorithm_name = algorithm_name
        self.optimize_mode = OptimizeMode(optimize_mode)
        self.json = None
        self.total_data = {}
        self.verbose = verbose
        self.model_count = 0

        self.bo = BayesianOptimizer(self, self.t_min, self.optimize_mode, self.beta)
        self.training_queue = []
        self.descriptors = []
        self.history = []

        self.max_model_size = max_model_size
        self.default_model_len = default_model_len
        self.default_model_width = default_model_width
        self.init_model_dir = []
        if os.path.isdir(init_model_dir):
            for parent, dirnames,model_files in os.walk(init_model_dir):
                if model_files is not None:
                    for model_file_name in model_files:
                        model_file=os.path.join(parent, model_file_name)
                        self.init_model_dir.append(model_file)
                else:
                    self.init_model_dir.append(0)
        elif os.path.isfile(init_model_dir):
            self.init_model_dir.append(init_model_dir)
        else:
            self.init_model_dir.append(0)
        self.search_space = dict()

    def update_search_space(self, search_space):
        """
        Update search space definition in tuner by search_space in neural architecture.
        """
        self.search_space = search_space

    def set_descriptors(self, model_id, generated_graph):
        self.descriptors[model_id] = generated_graph.extract_descriptor()

    def fake_generate_parameters(self, parameter_id, **kwargs):
        """
        Returns a initialized model.
        """
        self.init_search()

        new_father_id = None
        generated_graph = None

        graph, father_id, model_id = self.training_queue.pop(0)

        # from graph to json
        json_model_path = os.path.join(self.path, str(model_id) + ".json")
        json_out = graph_to_json(graph, json_model_path)
        self.total_data[parameter_id] = (json_out, father_id, model_id)

        return json_out

    def generate_parameters(self, parameter_id, **kwargs):
        """
        Returns a set of trial neural architecture, as a serializable object.

        Parameters
        ----------
        parameter_id : int
        """
        #If there is no history, slave node will use the fake model.
        if not self.history:
            print("If there is no history, generate_parameters should not be called!")
            exit(1)
        total_start=time.time()
        rate = 1

        if (os.path.exists(os.environ["AIPERF_WORKDIR"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/generate_time") and os.path.exists(os.environ["AIPERF_WORKDIR"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/train_time")):
            with open(os.environ["AIPERF_WORKDIR"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/generate_time", "r") as f:
                generate_time = float(f.read())
            with open(os.environ["AIPERF_WORKDIR"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/train_time", "r") as f:
                train_time = float(f.read())
            if (generate_time != 0) and (train_time != 0):
                realrate = int(train_time / generate_time)
                if (realrate < 5) and (realrate > 1):
                    rate = int(realrate)
                if (realrate <= 1):
                    rate = 1

        for i in range(rate):
            start=time.time()
            new_father_id = None
            generated_graph = None
            if not self.training_queue:
                new_father_id, generated_graph = self.generate()
                father_id,json_out,new_model_id = self.total_data[parameter_id]
                self.training_queue.append((generated_graph, new_father_id, new_model_id))
                #self.descriptors.append(generated_graph.extract_descriptor())
            else:
                print("training_queue should be an empty list.")
                exit(1)

            graph, father_id, model_id = self.training_queue.pop(0)
        # from graph to json
            json_model_path = os.path.join(self.path, str(model_id) + ".json")
            json_out = graph_to_json(graph, json_model_path)
            end=time.time()
        #self.total_data[parameter_id] = (json_out, father_id, model_id)
            json_and_id="json_out="+str(json_out)+"+father_id="+str(father_id)+"+parameter_id="+str(parameter_id)+"+history="+"True"
            lock.acquire()
            with open(os.environ["AIPERF_WORKDIR"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/trials/" + str(nni.get_trial_id()) + "/output.log","a+") as f:
                f.write("single_generate=" + str(end - start)+"\n")

            with open(os.environ["AIPERF_WORKDIR"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/graph.txt","a+") as f:
                f.write(json_and_id+"\n")
            lock.release()
        total_end=time.time()
        lock.acquire()
        with open(os.environ["AIPERF_WORKDIR"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/trials/" + str(nni.get_trial_id()) + "/output.log","a+") as f:
            f.write("total_generate=" + str(total_end - total_start)+"\n")
        lock.release()

        totime = total_end - total_start
        if totime<0:
            totime = 0-totime

        with open (os.environ["AIPERF_WORKDIR"] + "/mountdir/nni/experiments/" + str(nni.get_experiment_id()) + "/generate_time","w+") as f:
            gt = totime/rate
            f.write(str(gt))

        #return json_out, father_id

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """ Record an observation of the objective function.

        Parameters
        ----------
        parameter_id : int
        parameters : dict
        value : dict/float
            if value is dict, it should have "default" key.
        """
        reward = extract_scalar_reward(value)

        if parameter_id not in self.total_data:
            raise RuntimeError("Received parameter_id not in total_data.")

        (_, father_id, model_id) = self.total_data[parameter_id]
        
        graph = self.bo.searcher.load_model_by_id(model_id)
        # to use the value and graph
        self.add_model(reward, model_id)
        self.update(father_id, graph, reward, model_id)

    def init_search(self):
        """Call the generators to generate the initial architectures for the search."""
        import yaml
        trial_concurrency = int(os.popen('cat '+os.environ['AIPERF_WORKDIR']+'/trial_concurrency.txt').read().strip())
        if trial_concurrency > self.model_count: #判断当前训练的trial是否已超过第一轮trials
            #若没有超过第一轮trial，则判断当前的trial是否超过预生成的模型序列，若未超过，则正常设置num=count
            # if  len(self.init_model_dir) > self.model_count:
            #     num= self.model_count
            # else:
            #     #若预生成的序列元素少于节点数（第一轮trial个数），则trial重复读取modle，采用取余方式
            num = self.model_count % len(self.init_model_dir)
        else:
            num=0
        for generator in self.generators:
            if os.path.isfile(self.init_model_dir[num]):
                graph = generator(self.n_classes, self.input_shape).pre_generate(
                    self.init_model_dir[num]
                )
            else:
                graph = generator(self.n_classes, self.input_shape).generate(
                    self.default_model_len, self.default_model_width
                    )
            model_id = self.model_count
            self.model_count += 1
            self.training_queue.append((graph, -1, model_id))
            self.descriptors.append(graph.extract_descriptor())
            # add acc fake
            if trial_concurrency > model_id:
                self.bo.fit([graph.extract_descriptor()], [0.05+model_id*0.01])
            #    self.bo.add_child(-1, model_id)
                self.history.append( {"model_id": model_id, "metric_value": 0.05+model_id*0.01})
                if model_id == trial_concurrency-1 :
                    if len(self.init_model_dir) >= trial_concurrency:
                        for i in range(trial_concurrency):
                            self.bo.add_child(i-1, i)
                    else:
                        #按正常的model id构建search tree
                        for i in range(len(self.init_model_dir)):
                            self.bo.add_child(i-1, i)
                        #把尾部超出预生成模型路径的model id 取余后加至对应的father id上
                        for i in range(trial_concurrency-len(self.init_model_dir)):
                            j=(model_id-trial_concurrency+len(self.init_model_dir)+i+1) % len(self.init_model_dir)
                            self.bo.add_child(j ,i + len(self.init_model_dir))
                    ret_tree=self.bo.search_tree.get_dict(0)


    def generate(self):
        """Generate the next neural architecture.

        Returns
        -------
        other_info: any object
            Anything to be saved in the training queue together with the architecture.
        generated_graph: Graph
            An instance of Graph.
        """
        generated_graph, new_father_id = self.bo.generate(self.descriptors)
        if new_father_id is None:
            new_father_id = 0
            generated_graph = self.generators[0](
                self.n_classes, self.input_shape
            ).generate(self.default_model_len, self.default_model_width)

        return new_father_id, generated_graph

    def update(self, other_info, graph, metric_value, model_id):
        """ Update the controller with evaluation result of a neural architecture.

        Parameters
        ----------
        other_info: any object
            In our case it is the father ID in the search tree.
        graph: Graph
            An instance of Graph. The trained neural architecture.
        metric_value: float
            The final evaluated metric value.
        model_id: int
        """
        father_id = other_info
        t1 = time.time()
        self.bo.fit([graph.extract_descriptor()], [metric_value])
        trial_concurrency = int(os.popen('cat '+os.environ['AIPERF_WORKDIR']+'/trial_concurrency.txt').read().strip())
        if model_id >= trial_concurrency :
            self.bo.add_child(father_id, model_id)
        ret_tree=self.bo.search_tree.get_dict(0)
        t2 = time.time()
        print("Update time = " + str(t2 - t1))

    def add_model(self, metric_value, model_id):
        """ Add model to the history, x_queue and y_queue

        Parameters
        ----------
        metric_value : float
        graph : dict
        model_id : int

        Returns
        -------
        model : dict
        """
        
        # Update best_model text file
        ret = {"model_id": model_id, "metric_value": metric_value}
        trial_concurrency = int(os.popen('cat '+os.environ['AIPERF_WORKDIR']+'/trial_concurrency.txt').read().strip())
        if model_id < trial_concurrency:
            for i in range(len(self.history)):
                if self.history[i]['model_id'] == model_id:
                    self.history[i]['metric_value'] = metric_value
                    break
        else:
            self.history.append(ret)
        if model_id == self.get_best_model_id():
            file = open(os.path.join(self.path, "best_model.txt"), "w")
            file.write("best model: " + str(model_id))
            file.close()
        return ret

    def get_best_model_id(self):
        """ Get the best model_id from history using the metric value
        """

        if self.optimize_mode is OptimizeMode.Maximize:
            return max(self.history, key=lambda x: x["metric_value"])["model_id"]
        return min(self.history, key=lambda x: x["metric_value"])["model_id"]

    def load_model_by_id(self, model_id):
        """Get the model by model_id

        Parameters
        ----------
        model_id : int
            model index

        Returns
        -------
        load_model : Graph
            the model graph representation
        """

        with open(os.path.join(self.path, str(model_id) + ".json")) as fin:
            json_str = fin.read().replace("\n", "")

        load_model = json_to_graph(json_str)
        return load_model

    def load_best_model(self):
        """ Get the best model by model id

        Returns
        -------
        load_model : Graph
            the model graph representation
        """
        return self.load_model_by_id(self.get_best_model_id())

    def get_metric_value_by_id(self, model_id):
        """ Get the model metric valud by its model_id

        Parameters
        ----------
        model_id : int
            model index

        Returns
        -------
        float
             the model metric
        """
        for item in self.history:
            if item["model_id"] == model_id:
                return item["metric_value"]
        return None

    def import_data(self, data):
        pass
