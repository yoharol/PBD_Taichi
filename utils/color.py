def color_hex2tuple(hexcolor):
  b = hexcolor & 0xff
  hexcolor = hexcolor >> 8
  g = hexcolor & 0xff
  hexcolor = hexcolor >> 8
  r = hexcolor & 0xff
  return (r / 255.0, g / 255.0, b / 255.0)


white = color_hex2tuple(0xffffff)
black = color_hex2tuple(0x001219)
deep_blue = color_hex2tuple(0x005f73)
blue = color_hex2tuple(0x0a9396)
light_blue = color_hex2tuple(0x94d2bd)
light_yellow = color_hex2tuple(0xe9d8a6)
yellow = color_hex2tuple(0xee9b00)
orange = color_hex2tuple(0xca6702)
deep_orange = color_hex2tuple(0xbb3e03)
red = color_hex2tuple(0xae2012)
deep_red = color_hex2tuple(0x9b2226)
