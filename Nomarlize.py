import torch
import numpy as np

def normalize_cpt_minmax(data):
    return data


def normalize_parameters_lower(data):
    data = (data - torch.tensor([6.028646832674804, 0.06118921382433419, 15.000569971919921], dtype=torch.double)) \
           / torch.tensor([59.90211541756136 - 6.028646832674804,\
                           2.315424541615237 - 0.06118921382433419,\
                           29.999994038964836 - 15.000569971919921],
                          dtype=torch.double)
    return data


def normalize_parameters_upper(data):
    data = (data - torch.tensor([12.129299616830263, 0.9806484227796798, 15.000245839070825], dtype=torch.double)) \
           / torch.tensor([149.66360455034103 - 12.129299616830263,\
                           90.85003361434994 - 0.9806484227796798,\
                           29.99999742016262 - 15.000245839070825],
                          dtype=torch.double)
    return data


                                        #################################
                                        #################################
def normalize_input(data):
    data = (data - torch.tensor([6.0750843378958095, 0.06281373259381724, 15.006308017348147], dtype=torch.double)) \
           / torch.tensor([148.1308752071033 - 6.0750843378958095,\
                           89.27290732217779 -0.06281373259381724,\
                           29.997327455651842 - 15.006308017348147],
                          dtype=torch.double)
    return data

def normalize_output(data):
    return data

def normalize_output_log(data):
    return torch.log(data)

def unnormalize_output_log(data):
    return torch.exp(data)


def normalize_output_power(data):
    return torch.pow(data, 0.25)

def unnormalize_output_power(data):
    return torch.pow(data, 4)


def normalize_parameters_upper(data):
    data = (data - torch.tensor([12.129299616830263, 0.9806484227796798, 15.000245839070825], dtype=torch.double)) \
           / torch.tensor([149.66360455034103 - 12.129299616830263,\
                           90.85003361434994 - 0.9806484227796798,\
                           29.99999742016262 - 15.000245839070825],
                          dtype=torch.double)
