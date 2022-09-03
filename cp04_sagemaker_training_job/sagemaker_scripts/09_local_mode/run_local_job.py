import boto3
import sagemaker
from sagemaker.estimator import Estimator
  
def main():
    boto_session = boto3.Session(profile_name="develop")
     
    sagemaker_session = sagemaker.LocalSession(boto_session=boto_session)
    role = sagemaker_session.get_caller_identity_arn()
  
    estimator = Estimator(image_uri="sagemaker.gpu:latest",
                        entry_point="09_cifar.py",
                        role=role,
                        instance_count=1,
                        instance_type="local_gpu", # cpuならlocalにする
                        output_path="s3://koshian-bucket/sagemaker_local_logs/",
                        code_location="s3://koshian-bucket/sagemaker_local_logs/",
                        sagemaker_session=sagemaker_session,
                        base_job_name="local-cifar",
                        hyperparameters={
                            "ckpt_dir": "/opt/ml/model"
                        })
    estimator.fit()
 
if __name__ == "__main__":
    main()