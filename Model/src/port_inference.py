import os
from inference_sdk import InferenceHTTPClient

IMAGE_FOLDER_PATH = "../../Data/test" 
VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp") 

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="LlDMSrK66qbTQVAkd9C8"
)

if not os.path.exists(IMAGE_FOLDER_PATH):
    print(f"Error: 폴더를 찾을 수 없습니다 - {IMAGE_FOLDER_PATH}")
else:
    for filename in os.listdir(IMAGE_FOLDER_PATH):
        if filename.lower().endswith(VALID_EXTENSIONS):
            full_file_path = os.path.join(IMAGE_FOLDER_PATH, filename)
            
            print(f"🔍 Running inference on: {filename}")

            try:
                result = client.run_workflow(
                    workspace_name="ppe-aolqt",
                    workflow_id="custom-workflow-2",
                    images={
                        "image": full_file_path  
                    },
                    use_cache=True 
                )

                print(result) 
                print("-" * 30) 

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")

print("✅ All processing complete.")