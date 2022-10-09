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


def wrap_and_convert3(model, input_specs, output_model_dir=None):
    serving_signature = "serving_default"
    if not output_model_dir:

        @tf.function
        def model_body(model_inputs):
            return model(model_inputs)

        @tf.function(input_signature=input_specs)
        def model_func(inp1, inp2, inp3):
            return model_body([inp1, inp2, inp3])

        input_model_dir = tempfile.mkdtemp()
        output_model_dir = tempfile.mkdtemp()

        print("i/o dirs: ", input_model_dir, output_model_dir)

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

    def single_batch_function(func_inputs3):
        inp1, inp2, inp3 = [
                tf.cast(
                    tf.convert_to_tensor(arg),
                    dtype=input_specs[i].dtype
                ) for i, arg in enumerate(func_inputs3)
        ]
        return serving_fn(
            **{
                input_specs[0].name: inp1, 
                input_specs[1].name: inp2,
                input_specs[2].name: inp3,
            }
        )["output_0"]

    return single_batch_function


def wrap_and_convert2(model, input_specs, output_model_dir=None):
    serving_signature = "serving_default"
    if not output_model_dir:

        @tf.function
        def model_body(model_inputs):
            return model(model_inputs)

        @tf.function(input_signature=input_specs)
        def model_func(inp1, inp2):
            return model_body([inp1, inp2])

        input_model_dir = tempfile.mkdtemp()
        output_model_dir = tempfile.mkdtemp()

        print("i/o dirs: ", input_model_dir, output_model_dir)

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

    def single_batch_function(func_inputs2):
        inp1, inp2 = [tf.convert_to_tensor(i) for i in func_inputs2]
        # inp1_e, inp2_e = [tf.expand_dims(i, axis=0) for i in (inp1, inp2)]
        return serving_fn(
            **{
                input_specs[0].name: inp1, 
                input_specs[1].name: inp2,
            }
        )["output_0"]

    return single_batch_function
