
# 0.0.2

### Dev
- Dev instructions & bootstrap script for easier start devving
- Added typer tasks for easier dev setup


### Docker
- Changed api volumes to be the 3 dir's it needs, so it doesn't pull gitignore stuff (such as .venv's)
- Only expose redis on localhost (digital ocean bricks an account if you expose redis on the host)
- Added API Key Header so auto doc builders (e.g. swagger) let you use api keys. Removed the unused allow_api_key flag.
- Added various notes to api key utils

### Users
- Added fingerprint hash
- Restructed event listeners to enforce order
- Coldkey optional - defaults to payment address
- Hotkey optional - defaults to ZERO_HOTKEY [NOTE: need to handle / check for this better]
- Added username validation on the register endpoint

### Auth
- Added nonce and signature check to API endpoints - to ensure that hotkeys can only be assigned to the real owner of the key
- Add validation to hotkeys, to ensure the user actually owns the hotkey
- Enforced single user per hotkey


### Generic
- Util function for nonce checking