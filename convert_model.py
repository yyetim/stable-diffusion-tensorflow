import tensorflow as tf
import tempfile
import subprocess

CONVERTER_IMAGE_NAME = "gcr.io/cloud-tpu-v2-images-dev/tpu_inference_converter_cli"

def wrap_model(model, model_inputs, input_specs):
    @tf.function(input_signatures=input_specs)
    def model_func(model_inputs):
        return model(model_inputs)

    with tempfile.TemporaryDirector() as d:
        signatures = {'serving_default': model_func.get_concrete_function()}
        save_options = tf.saved_model.SaveOptions(function_aliases={
            'tpu_func': model_func,
        })
        tf.saved_model.save(model, d, signatures, save_options)

        subprocess.check_output(
                ["docker", "run", CONVERTER_IMAGE_NAME]
        )
