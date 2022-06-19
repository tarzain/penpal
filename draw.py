from svgpathtools import wsvg, Line, QuadraticBezier, Path

from freetype import Face


def tuple_to_imag(t):
    return t[0] + t[1] * 1j


face = Face('./fonts/CNC_Fancy King.ttf')
face.set_char_size(2 * 3)
face.load_char('a')
outline = face.glyph.outline
y = [t[1] for t in outline.points]
# flip the points
outline_points = [(p[0], max(y) - p[1]) for p in outline.points]
start, end = 0, 0
paths = []

for i in range(len(outline.contours)):
    end = outline.contours[i]
    points = outline_points[start:end + 1]
    points.append(points[0])
    tags = outline.tags[start:end + 1]
    tags.append(tags[0])

    segments = [[points[0], ], ]
    for j in range(1, len(points)):
        segments[-1].append(points[j])
        if tags[j] and j < (len(points) - 1):
            segments.append([points[j], ])
    for segment in segments:
        if len(segment) == 2:
            paths.append(Line(start=tuple_to_imag(segment[0]),
                              end=tuple_to_imag(segment[1])))
        elif len(segment) == 3:
            paths.append(QuadraticBezier(start=tuple_to_imag(segment[0]),
                                         control=tuple_to_imag(segment[1]),
                                         end=tuple_to_imag(segment[2])))
        elif len(segment) == 4:
            C = ((segment[1][0] + segment[2][0]) / 2.0,
                 (segment[1][1] + segment[2][1]) / 2.0)

            paths.append(QuadraticBezier(start=tuple_to_imag(segment[0]),
                                         control=tuple_to_imag(segment[1]),
                                         end=tuple_to_imag(C)))
            paths.append(QuadraticBezier(start=tuple_to_imag(C),
                                         control=tuple_to_imag(segment[2]),
                                         end=tuple_to_imag(segment[3])))
        elif len(segment) == 5:
            C = ((segment[1][0] + segment[3][0]) / 2.0,
                (segment[1][1] + segment[3][1]) / 2.0)

            paths.append(QuadraticBezier(start=tuple_to_imag(segment[0]),
                                        control=tuple_to_imag(segment[1]),
                                        end=tuple_to_imag(C)))
            paths.append(QuadraticBezier(start=tuple_to_imag(C),
                                        control=tuple_to_imag(segment[3]),
                                        end=tuple_to_imag(segment[4])))
        elif len(segment) == 6:
            C12 = segment[1]
            C23 = segment[2]
            C34 = segment[3]
            C45 = segment[4]

            P1 = segment[0]
            P2 = ((segment[1][0] + segment[2][0]) / 2.0,
                 (segment[1][1] + segment[2][1]) / 2.0)
            P3 = ((segment[2][0] + segment[3][0]) / 2.0,
                 (segment[2][1] + segment[3][1]) / 2.0)
            P4 = ((segment[3][0] + segment[4][0]) / 2.0,
                 (segment[3][1] + segment[4][1]) / 2.0)
            P5 = segment[5]

            paths.append(QuadraticBezier(start=tuple_to_imag(P1),
                                         control=tuple_to_imag(C12),
                                         end=tuple_to_imag(P2)))
            paths.append(QuadraticBezier(start=tuple_to_imag(P2),
                                         control=tuple_to_imag(C23),
                                         end=tuple_to_imag(P3)))
            paths.append(QuadraticBezier(start=tuple_to_imag(P3),
                                         control=tuple_to_imag(C34),
                                         end=tuple_to_imag(P4)))
            paths.append(QuadraticBezier(start=tuple_to_imag(P4),
                                         control=tuple_to_imag(C45),
                                         end=tuple_to_imag(P5)))
        elif len(segment) > 6:
            C = ((segment[1][0] + segment[len(segment)-2][0]) / 2.0,
                (segment[1][1] + segment[len(segment)-2][1]) / 2.0)

            paths.append(QuadraticBezier(start=tuple_to_imag(segment[0]),
                                        control=tuple_to_imag(segment[1]),
                                        end=tuple_to_imag(C)))
            paths.append(QuadraticBezier(start=tuple_to_imag(C),
                                        control=tuple_to_imag(segment[len(segment)-2]),
                                        end=tuple_to_imag(segment[len(segment)-1])))
        else:
            print(f"incompatible segment length: {len(segment)}")
    start = end + 1

path = Path(*paths)
wsvg(path, filename="text.svg")


from pyaxidraw import axidraw

ad = axidraw.AxiDraw()
ad.plot_setup("text.svg")
ad.plot_run()