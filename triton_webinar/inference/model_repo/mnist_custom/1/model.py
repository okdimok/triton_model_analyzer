import triton_python_backend_utils as pb_utils
import json

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args["model_config"])

        # Get output configuration
        output_config = pb_utils.get_output_config_by_name(model_config, "output")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(
            output_config["data_type"]
        )

    def execute(self, requests):
        responses = []
        for request in requests:
            inTensor = pb_utils.get_input_tensor_by_name(request, "input")
            inference_request = pb_utils.InferenceRequest(
                model_name='mnist_trt',
                requested_output_names=['output'],
                inputs=[inTensor])
            
            inference_response = inference_request.exec()

            if inference_response.has_error():
                raise pb_utils.TritonModelException(
                    inference_response.error().message())
            else:
                # Extract the output tensors from the inference response.
                output = pb_utils.get_output_tensor_by_name(
                    inference_response, 'output')
                
                inference_response = pb_utils.InferenceResponse(
                    output_tensors=[output])
                responses.append(inference_response)
        
        return responses
          
    def finalize(self):
        print("Cleaning up...")
