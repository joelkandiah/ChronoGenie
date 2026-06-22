from pathlib import Path

import torch.nn as nn


def count_parameters(module):
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters())


def count_trainable_parameters(module):
    if module is None:
        return 0
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def _temporal_counts(temporal_model):
    if isinstance(temporal_model, nn.ModuleList):
        if len(temporal_model) == 0:
            return 0, 0, 0, 0

        train_params = sum(count_parameters(module) for module in temporal_model)
        trainable_params = sum(count_trainable_parameters(module) for module in temporal_model)
        test_params = count_parameters(temporal_model[0])
        test_trainable_params = count_trainable_parameters(temporal_model[0])
        return train_params, trainable_params, test_params, test_trainable_params

    train_params = count_parameters(temporal_model)
    trainable_params = count_trainable_parameters(temporal_model)
    return train_params, trainable_params, train_params, trainable_params


def write_model_features(output_dir, temporal_model, local_profile_encoder=None,
                         interaction_encoder=None, model_type="unknown",
                         file_name="model_features.txt"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    temporal_train_params, temporal_trainable_params, temporal_test_params, temporal_test_trainable_params = _temporal_counts(temporal_model)
    local_profile_params = count_parameters(local_profile_encoder)
    local_profile_trainable = count_trainable_parameters(local_profile_encoder)
    interaction_params = count_parameters(interaction_encoder)
    interaction_trainable = count_trainable_parameters(interaction_encoder)

    train_total_params = local_profile_params + interaction_params + temporal_train_params
    test_total_params = local_profile_params + interaction_params + temporal_test_params
    train_total_trainable = local_profile_trainable + interaction_trainable + temporal_trainable_params
    test_total_trainable = local_profile_trainable + interaction_trainable + temporal_test_trainable_params

    file_path = output_path / file_name
    lines = [
        "GENIE model features",
        f"model_type: {model_type}",
        f"multistep_heads: {len(temporal_model) if isinstance(temporal_model, nn.ModuleList) else 1}",
        f"local_profile_encoder_params: {local_profile_params}",
        f"local_profile_encoder_trainable_params: {local_profile_trainable}",
        f"interaction_encoder_params: {interaction_params}",
        f"interaction_encoder_trainable_params: {interaction_trainable}",
        f"temporal_train_params: {temporal_train_params}",
        f"temporal_trainable_params: {temporal_trainable_params}",
        f"temporal_test_params: {temporal_test_params}",
        f"temporal_test_trainable_params: {temporal_test_trainable_params}",
        f"train_total_params: {train_total_params}",
        f"test_total_params: {test_total_params}",
        f"train_total_trainable_params: {train_total_trainable}",
        f"test_total_trainable_params: {test_total_trainable}",
        f"train_parameter_memory_mb: {train_total_params * 4 / (1024 ** 2):.4f}",
        f"test_parameter_memory_mb: {test_total_params * 4 / (1024 ** 2):.4f}",
    ]

    with open(file_path, "w") as f:
        f.write("\n".join(lines) + "\n")
