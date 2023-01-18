import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np


def summary(model, dummy_batch_input, batch_size):
    result, params_info = summary_string(
        model, dummy_batch_input, batch_size)
    print(result)

    return params_info


def summary_string(model, dummy_batch_input, batch_size):
    # if dtypes == None:
    #     dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            # if not isinstance(input[0], dict):
            #     summary[m_key]["input_shape"] = list(input[0].size()) 
            #     summary[m_key]["input_shape"][0] = batch_size
            # else:
            #     summary[m_key]["input_shape"] = 0
            # if isinstance(output, (list, tuple)):
            #     summary[m_key]["output_shape"] = [
            #         [-1] + list(o.size())[1:] for o in output
            #     ]
            # else:
            #     summary[m_key]["output_shape"] = list(output.size())
            #     summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    x = dummy_batch_input

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>15}".format(
        "Layer (type)", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>15}".format(
            layer,
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        # total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    # total_input_size = abs(np.prod(sum(input_size, ()))
    #                        * batch_size * 4. / (1024 ** 2.))
    # total_output_size = abs(2. * total_output * 4. /
    #                         (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    # total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    # summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    # summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    # summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "Estimated size without input and output (MB): %0.2f" % total_params_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str, (total_params, trainable_params)
