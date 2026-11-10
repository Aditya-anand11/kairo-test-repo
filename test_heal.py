# Testing self healing
import boto3

def test_boto3():
  assert boto3.__version__ is not None
