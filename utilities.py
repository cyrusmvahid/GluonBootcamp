import random
class visual_utilities():
    @staticmethod
    def random_hex_colours(count):
        colours = []
        r = lambda: random.randint(0, 255)
        for i in range(count):
            colours.append('#%02X%02X%02X' % (r(), r(), r()))
        return colours


