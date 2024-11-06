
# 0.0.2

### Dev
- Dev instructions & bootstrap script for easier start devving
- Added invoke tasks for easy setup

### Docker
- Changed api volumes to be the 3 dir's it needs, so it doesn't pull gitignore stuff (such as .venv's)
- Only expose redis on localhost (digital ocean bricks an account if you expose redis on the host)
- Added API Key Header so auto doc builders (e.g. swagger) let you use api keys. Removed the unused allow_api_key flag.
- Added various notes to api key utils

