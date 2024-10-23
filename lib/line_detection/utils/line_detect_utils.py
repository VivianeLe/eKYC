def best_bbox(t):
    return t[t[:, -1].argsort()[-1]]


def polygon_from_corners(t):
    for i, p in enumerate(t):
        x, y, w, h = p[2], p[3], p[4], p[5]
        t[i][2], t[i][4] = (x - w / 2, x + w / 2)
        t[i][3], t[i][5] = (y - h / 2, y + h / 2)

    if len(t) > 0:
        address_1 = t[t[:, 1] == 8]
        address_2 = t[t[:, 1] == 9]
        birthday = t[t[:, 1] == 3]
        hometown_1 = t[t[:, 1] == 6]
        hometown_2 = t[t[:, 1] == 7]
        ids = t[t[:, 1] == 0]
        name = t[t[:, 1] == 2]
        nation = t[t[:, 1] == 5]
        sex = t[t[:, 1] == 4]
        passport_id = t[t[:, 1] == 1]

        address_1 = best_bbox(address_1)[2:6] if len(address_1) != 0 else address_1
        address_2 = best_bbox(address_2)[2:6] if len(address_2) != 0 else address_2
        birthday = best_bbox(birthday)[2:6] if len(birthday) != 0 else birthday
        hometown_1 = best_bbox(hometown_1)[2:6] if len(hometown_1) != 0 else hometown_1
        hometown_2 = best_bbox(hometown_2)[2:6] if len(hometown_2) != 0 else hometown_2
        ids = best_bbox(ids)[2:6] if len(ids) != 0 else ids
        name = best_bbox(name)[2:6] if len(name) != 0 else name
        nation = best_bbox(nation)[2:6] if len(nation) != 0 else nation
        sex = best_bbox(sex)[2:6] if len(sex) != 0 else sex
        passport_id = best_bbox(passport_id)[2:6] if len(passport_id) != 0 else passport_id

    else:
        return None

    return (address_1, address_2, birthday, hometown_1, hometown_2, ids, name, nation, sex, passport_id)


def increase_size_box(bbox, img_size, w_extend_size, h_extend_size):
    xmin, ymin, xmax, ymax = max(int(bbox[0]), 0), max(0, int(bbox[1])), \
                             min(int(bbox[2]), img_size[1]), min(int(bbox[3]), img_size[0])

    # Extend
    xmin_extend, ymin_extend, xmax_extend, ymax_extend = max(0, xmin - int(
        (xmax - xmin) * w_extend_size)), \
                                                         max(0, ymin - int(
                                                             (ymax - ymin) * h_extend_size)), \
                                                         xmax + int((xmax - xmin) * w_extend_size), \
                                                         ymax + int((ymax - ymin) * h_extend_size)

    bbox_extend = [xmin_extend, ymin_extend, xmax_extend, ymax_extend]
    return bbox_extend

def crop_img_from_bbox(img, box):
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    line_cropped = img.copy().crop((xmin, ymin, xmax, ymax))
    return line_cropped