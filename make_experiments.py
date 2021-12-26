import subprocess


features = [
    'has_twitter,twitter_followers',
    'has_twitter',
    'twitter_followers',
    'has_facebook',
]

for feature in features:
    subprocess.call(f"python train.py --task='fact' --features='{feature}'", shell=True)
    subprocess.call(f"python train.py --task='bias' --features='{feature}'", shell=True)
