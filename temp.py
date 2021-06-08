def check_iterable(val):
    try:
        val = iter(val)
        return True
    except:
        return False


def check_type(val1, val2):
    try:
        val2 = type(val1)(val2)
        return True
    except:
        return False


def change_type(val1, val2):
    if isinstance(val2, str):
        if not check_iterable(val1):
            return type(val1)(val2)
        else:
            if isinstance(val1, list):
                val2 = type(val1)(val2[1:-1].split(','))
                for i, item in enumerate(val2):
                    val2[i] = item.replace(" ", "")
    return type(val1)(val2)


def make_the_same(val1, val2):
    if not check_type(val1, val2):
        return val1
    val2 = change_type(val1, val2)
    if isinstance(val1, list):
        for i, item in enumerate(iter(val1)):
            val1[i] = make_the_same(val1[i], val2[i])
    else:
        val1 = change_type(val1, val2)
    return val1
