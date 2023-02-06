"""
给出一种颜色和颜色集，从颜色集中返回与该颜色最接近的颜色
"""
from colorsys import rgb_to_hsv

def to_hsv(color):

    """ converts color tuples to floats and then to hsv """

    return rgb_to_hsv(*[x/255.0 for x in color]) #rgb_to_hsv wants floats!

def color_dist(c1, c2):

    """ returns the squared euklidian distance between two color vectors in hsv space """

    return sum( (a-b)**2 for a,b in zip(to_hsv(c1),to_hsv(c2)) )

def min_color_diff( color_to_match, colors):

    """ returns the `(distance, color_name)` with the minimal distance to `colors`"""

    return min( # overal best is the best match to any color:

        (color_dist(color_to_match, color), index) # (distance to `test` color, color name)

        for index, color in enumerate(colors))