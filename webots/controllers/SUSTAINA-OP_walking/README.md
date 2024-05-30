
# SUSTAINA-OP_walking

Webots controller for the SUSTAINA-OP proto model.  
This repository includes both a walking pattern generator and a walking target command sending api.

---

## 👀Emvironment
- webots R2023b
- python3.8 or later
---

## ⚙️Installation

Install required packages using pip.

```bash
  pip3 install -r requirements.txt
```

---  
## 🧬Features

- Sending walking direction by zmq and protobuf message.
- Get image data from camera.
- Viewing image from camera.

---  

## ⚡️Example

### CLI

#### **Sending walking direction.**

```bash
python3 walk_client.py <target x> <target y> <target theta>
```


#### **Viewing image from camera**
```bash
python3 view_image.py
```

### Python
    
#### Sending walking direction command from python script and receiving camera image data.

```python
import walk_client

client = walk_client.WalkClient()

# Send walking direction.
client.sendCommand(0.5, 0.0, 0.3) # x, y, theta

# Receive image data from camera.
# This will return protobuf message(https://github.com/SUSTAINA-OP/SUSTAINA-OP_Webots/blob/master/webots/controllers/SUSTAINA-OP_walking/walk_command.proto). 
image = client.getImage() 

```

--- 

## 🧾License

[apache license 2.0](https://www.apache.org/licenses/LICENSE-2.0)
