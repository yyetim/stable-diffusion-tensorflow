import os
import tensorflow as tf
import tempfile
import subprocess

CONVERTER_IMAGE_NAME = "gcr.io/cloud-tpu-v2-images-dev/tpu_inference_converter_cli"
CONVERTER_OPTIONS = """
tpu_functions {
  function_alias: "tpu_func"
}
"""


def run_docker_command(input_dir, output_dir):
    docker_output = subprocess.check_output(
        [
            "docker",
            "run",
            "--mount",
            f"type=bind,source={input_dir},target=/tmp/input,readonly",
            "--mount",
            f"type=bind,source={output_dir},target=/tmp/output",
            "gcr.io/cloud-tpu-v2-images-dev/tpu_inference_converter_cli:cl_474604298",
            "--input_model_dir=/tmp/input",
            "--output_model_dir=/tmp/output",
            '--converter_options_string=tpu_functions { function_alias: "tpu_func"}',
        ]
    )
    print(docker_output)
    sudo_output = subprocess.check_output(
        ["sudo", "chown", "-R", os.getenv("USER"), f"{output_dir}"]
    )
    print(sudo_output)


def as_tf_function_with_unpacked_args(model_body, input_specs):
    num_args = len(input_specs)
    if num_args == 1:

        @tf.function(input_signature=input_specs)
        def model_func(inp1):
            return model_body([inp1])

    elif num_args == 2:

        @tf.function(input_signature=input_specs)
        def model_func(inp1, inp2):
            return model_body([inp1, inp2])

    elif num_args == 3:

        @tf.function(input_signature=input_specs)
        def model_func(inp1, inp2, inp3):
            return model_body([inp1, inp2, inp3])

    else:
        raise ValueError(f"Unimplemented num args in input_specs: {input_specs}")
    return model_func


class ConvertedModel:
    def __init__(self, input_specs, serving_fn):
        self.input_specs = input_specs
        self.serving_fn = serving_fn

    def predict_on_batch(self, func_inputs):
        tensor_inputs = [
            tf.cast(tf.convert_to_tensor(arg), dtype=self.input_specs[i].dtype)
            for i, arg in enumerate(func_inputs)
        ]
        serving_function_inputs = {
            input_spec.name: tensor_input
            for input_spec, tensor_input in zip(self.input_specs, tensor_inputs)
        }
        return self.serving_fn(**serving_function_inputs)["output_0"]


def wrap_and_convert(model, input_specs, output_model_dir=None):
    serving_signature = "serving_default"
    if not output_model_dir:

        @tf.function
        def model_body(model_inputs):
            return model(model_inputs)

        input_model_dir = tempfile.mkdtemp()
        output_model_dir = tempfile.mkdtemp()

        print("i/o dirs: ", input_model_dir, output_model_dir)

        model_func = as_tf_function_with_unpacked_args(model_body, input_specs)
        signatures = {serving_signature: model_func.get_concrete_function()}
        save_options = tf.saved_model.SaveOptions(
            function_aliases={
                "tpu_func": model_func,
            }
        )
        tf.saved_model.save(model, input_model_dir, signatures, save_options)

        command_out = run_docker_command(input_model_dir, output_model_dir)
        print(command_out)

    new_model = tf.saved_model.load(output_model_dir)
    serving_fn = new_model.signatures[serving_signature]

    return ConvertedModel(input_specs, serving_fn)
