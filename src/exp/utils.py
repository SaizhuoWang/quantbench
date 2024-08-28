def get_machine():
    import os
    hostname = os.uname()[1]
    if 'finai' in hostname or 'hgx' in hostname:
        return 'hgx'
    else:
        return 'dgx'