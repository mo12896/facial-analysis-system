"""
 * Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose
 *
 * @author Danil Osokin
 *
 * Includes code from Danil Osokin on https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch under Apache license
 * Copyright (c) 2019 Danil Osokin
 *
 * Created at : 2022-11-25
"""

import collections


def load_state(net, checkpoint):
    source_state = checkpoint["state_dict"]
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        if (
            target_key in source_state
            and source_state[target_key].size() == target_state[target_key].size()
        ):
            new_target_state[target_key] = source_state[target_key]
        else:
            new_target_state[target_key] = target_state[target_key]
            print(
                "[WARNING] Not found pre-trained parameters for {}".format(target_key)
            )

    net.load_state_dict(new_target_state)


def load_from_mobilenet(net, checkpoint):
    source_state = checkpoint["state_dict"]
    target_state = net.state_dict()
    new_target_state = collections.OrderedDict()
    for target_key, target_value in target_state.items():
        k = target_key
        if k.find("model") != -1:
            k = k.replace("model", "module.model")
        if (
            k in source_state
            and source_state[k].size() == target_state[target_key].size()
        ):
            new_target_state[target_key] = source_state[k]
        else:
            new_target_state[target_key] = target_state[target_key]
            print(
                "[WARNING] Not found pre-trained parameters for {}".format(target_key)
            )

    net.load_state_dict(new_target_state)
