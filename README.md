# MyWaymo-Object_Detection


<img width="943" alt="Image" src="https://github.com/user-attachments/assets/af08432a-9116-4e46-906c-82a99808199f" />

## Download San Francisco Car Class Detection Bounding Box Exmaple Zip File Image
[example.png.zip](https://github.com/user-attachments/files/19377235/example.png.zip)


## Create tf records

```bash
python convert.py -p /home/workspace/training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.
```

## Train the model
```bash
python training.py --imdir GTSRB/Final_Training/Images/
```