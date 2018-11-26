# KTP OCR

## Requirements
OpenCV2, Numpy, Pandas and Google Cloud python library

## Running it
You need to put your google app credential on this line
```
# import google app credential
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cfg.google_app_credential
```
I'm using cfg, but you're free to use anything.

The minimum code would be,

```
# create api object
api = VisionAPIOCRLocationBased(image)

# get text
text = api.get_text()
```