import tensorflow as tf
import tempfile
import subprocess

CONVERTER_IMAGE_NAME = "gcr.io/cloud-tpu-v2-images-dev/tpu_inference_converter_cli"
CONVERTER_OPTIONS = """
tpu_functions {
  function_alias: "tpu_func"
}
"""

def make_docker_command(input_dir, output_dir):
    return [
        'docker', 'run', '--mount',
        f'type=bind,source={input_dir},target=/tmp/input,readonly',
        '--mount',
        f'type=bind,source={output_dir},target=/tmp/output',
        'gcr.io/cloud-tpu-v2-images-dev/tpu_inference_converter_cli:cl_474604298',
        '--input_model_dir=/tmp/input', '--output_model_dir=/tmp/output',
        '--converter_options_string=\'tpu_functions { function_alias: "tpu_func"}\''
    ]

def wrap_and_convert3(model, input_specs, output_model_dir=None):
    if not output_model_dir:
        @tf.function(input_signature=input_specs)
        def model_func(inp1, inp2, inp3):
            return model([inp1, inp2, inp3])

        input_model_dir = tempfile.mkdtemp()
        output_model_dir = tempfile.mkdtemp()

        print("i/o dirs: ", input_model_dir, output_model_dir)

        signatures = {'serving_default': model_func.get_concrete_function()}
        save_options = tf.saved_model.SaveOptions(function_aliases={
            'tpu_func': model_func,
        })
        tf.saved_model.save(model, input_model_dir, signatures, save_options)

        command_out = subprocess.check_output(
            make_docker_command(input_model_dir, output_model_dir)
        )
        print(command_out)

    new_model = tf.saved_model.load(output_model_dir)
    serving_signature = 'serving_default'
    serving_fn = model.signatures[serving_signature]

    def single_batch_function(inp1, inp2, inp3):
        inp1_e, inp2_e, inp3_e = [tf.expand_dims(i, axis=0) for i in (inp1, inp2, inp3)]
        return serving_fn(inp1_e, inp2_e, inp3_e)
        

    return single_batch_function
