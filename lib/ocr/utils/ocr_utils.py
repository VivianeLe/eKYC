import time

import editdistance
import re

def remove_accent(text):
    accent_seq = 'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    no_accent_seq = 'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAaEeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    assert  len(accent_seq) == len(no_accent_seq), "Length accent_seq must equal " \
                                                   "length no_accent_seq: {} != {}".format(len(accent_seq), len(no_accent_seq))
    no_accent_text = ""
    for char in text:
        if char in accent_seq:
            no_accent_text += no_accent_seq[accent_seq.index(char)]
        else:
            no_accent_text += char
    return no_accent_text

def normalize_field(field_str, lower=True, noaccent=True, no_space=True, no_special_char=True):
    if lower:
        field_str = field_str.lower()
    if noaccent:
        field_str = remove_accent(field_str)
    if no_space:
        field_str = field_str.replace(" ", "")
    if no_special_char:
        field_str = re.sub(r'[^A-Za-z]', "", field_str)

    return field_str

def check_need_normalize(field_str, type='sex'):
    is_need_norm = False
    if type == 'sex':
        if field_str != 'Nam' and field_str != 'Nữ' and field_str != 'NAM/M' and field_str != 'NỮ/F':
            is_need_norm = True
    elif type == 'nation':
        if field_str != 'Việt Nam/VIETNAMESE' and field_str != 'Việt Nam':
            is_need_norm = True

    return is_need_norm
def check_is_sex_passport(sex_str):
    ratio_upper = cal_ratio_upper(sex_str)
    if ratio_upper >= 0.6:
        return True
    else:
        return False

def get_distance(word_1, word_2):
    return editdistance.eval(word_1, word_2)

def processing_sex(result_dict):
    if 'sex' in result_dict:
        sex_field = result_dict['sex']
        if isinstance(sex_field, list):
            sex_field = sex_field[0]

        if check_need_normalize(sex_field, type='sex'):
            is_sex_passport = check_is_sex_passport(sex_field)
            if is_sex_passport:
                sex_normalized = normalize_field(sex_field, lower=True, noaccent=True, no_space=True,
                                                 no_special_char=False)
                dist_male = get_distance(sex_normalized, 'nam/n')
                dist_female = get_distance(sex_normalized, 'nu/f')
                if dist_male < dist_female or (dist_male == dist_female and len(sex_normalized) >= 5):
                    result_dict['sex'] = ["NAM/M"]
                else:
                    result_dict['sex'] = ["NỮ/F"]
            else:
                sex_normalized = normalize_field(sex_field, lower=True, noaccent=True, no_space=True, no_special_char=True)
                dist_male = get_distance(sex_normalized, 'nam')
                dist_female = get_distance(sex_normalized, 'nu')

                if dist_male < dist_female or (dist_male == dist_female and len(sex_normalized) >= 3):
                    result_dict['sex'] = ["Nam"]
                else:
                    result_dict['sex'] = ["Nữ"]

    return result_dict

def cal_ratio_upper(text):
    text = text.replace(" ", "")
    num_upper = 0
    for char in text:
        if char.isupper():
            num_upper += 1
    ratio = num_upper / len(text)
    return ratio

def check_field_is_long_nation(nation_normalized, nation_non_normalize):
    if nation_normalized.count('viet') == 2 or (nation_normalized.count('viet') == 1 and cal_ratio_upper(nation_non_normalize) >= 0.6):
        return True
    else:
        return False


def processing_nation(result_dict):
    if 'nation' in result_dict:
        nation_field = result_dict['nation']
        if isinstance(nation_field, list):
            nation_field = nation_field[0]
        nation_non_normalize = nation_field
        if check_need_normalize(nation_field, type='nation'):
            nation_normalized = normalize_field(nation_field, lower=True, noaccent=True,
                                                no_space=True, no_special_char=True)

            if check_field_is_long_nation(nation_normalized, nation_non_normalize):
                result_dict['nation'] = ["VIỆT NAM/VIETNAMESE"]
            elif nation_normalized.count('viet') == 1:
                result_dict['nation'] = ["Việt Nam"]

    return result_dict

def processing_name(result_dict):
    if 'name' in result_dict:
        name_field = result_dict['name']
        if isinstance(name_field, list):
            name_field = name_field[0]
        result_dict['name'] = [name_field.upper()]

    return result_dict

def post_processing_result(result_dict, norm_sex=True, norm_nation=True, norm_name=True):
    if norm_sex:
        result_dict = processing_sex(result_dict)
    if norm_nation:
        result_dict = processing_nation(result_dict)
    if norm_name:
        result_dict = processing_name(result_dict)
    return result_dict


if __name__ == "__main__":
    t1 = time.time()
    for i in range(1):
        print(post_processing_result({'id': '030099005368', 'name': 'NGUYỄN ĐỨC LỘC', 'birthday': '13/11/1999', 'sex': 'Namf', 'nation': 'Việt Nam/VIETNAMESEE', 'hometown': '', 'address': 'Cẩm Văn, Cẩm Giàng Hải Dương'}))
    print((time.time() - t1) / 1000)






