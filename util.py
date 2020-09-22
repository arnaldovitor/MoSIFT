import math

cells = [15, 31, 47, 63, 79, 95, 111, 127, 143, 159, 175, 191, 207, 223, 239, 255]


def gen_neighbors(x, y):
    neighbor_1 = [[x - 8, y + 8], [x - 7, y + 8], [x - 6, y + 8], [x - 5, y + 8],
          [x - 8, y + 7], [x - 7, y + 7], [x - 6, y + 7], [x - 5, y + 7],
          [x - 8, y + 6], [x - 7, y + 6], [x - 6, y + 6], [x - 5, y + 6],
          [x - 8, y + 5], [x - 7, y + 5], [x - 6, y + 5], [x - 5, y + 5]]

    neighbor_2 = [[x - 4, y + 8], [x - 3, y + 8], [x - 2, y + 8], [x - 1, y + 8],
          [x - 4, y + 7], [x - 3, y + 7], [x - 2, y + 7], [x - 1, y + 7],
          [x - 4, y + 6], [x - 3, y + 6], [x - 2, y + 6], [x - 1, y + 6],
          [x - 4, y + 5], [x - 3, y + 5], [x - 2, y + 5], [x - 1, y + 5]]

    neighbor_3 = [[x + 1, y + 8], [x + 2, y + 8], [x + 3, y + 8], [x + 4, y + 8],
          [x + 1, y + 7], [x + 2, y + 7], [x + 3, y + 7], [x + 4, y + 7],
          [x + 1, y + 6], [x + 2, y + 6], [x + 3, y + 6], [x + 4, y + 6],
          [x + 1, y + 5], [x + 2, y + 5], [x + 3, y + 5], [x +  4, y + 5]]

    neighbor_4 = [[x + 5, y + 8], [x + 6, y + 8], [x + 7, y + 8], [x + 8, y + 8],
          [x + 5, y + 7], [x + 6, y + 7], [x + 7, y + 7], [x + 8, y + 7],
          [x + 5, y + 6], [x + 6, y + 6], [x + 7, y + 6], [x + 8, y + 6],
          [x + 5, y + 5], [x + 6, y + 5], [x + 7, y + 5], [x + 8, y + 5]]

    neighbor_5 = [[x - 8, y + 4], [x - 7, y + 4], [x - 6, y + 4], [x - 5, y + 4],
          [x - 8, y + 3], [x - 7, y + 3], [x - 6, y + 3], [x - 5, y + 3],
          [x - 8, y + 2], [x - 7, y + 2], [x - 6, y + 2], [x - 5, y + 2],
          [x - 8, y + 1], [x - 7, y + 1], [x - 6, y + 1], [x - 5, y + 1]]

    neighbor_6 = [[x - 4, y + 4], [x - 3, y + 4], [x - 2, y + 4], [x - 1, y + 4],
          [x - 4, y + 3], [x - 3, y + 3], [x - 2, y + 3], [x - 1, y + 3],
          [x - 4, y + 2], [x - 3, y + 2], [x - 2, y + 2], [x - 1, y + 2],
          [x - 4, y + 1], [x - 3, y + 1], [x - 2, y + 1], [x - 1, y + 1]]

    neighbor_7 = [[x + 1, y + 4], [x + 2, y + 4], [x + 3, y + 4], [x + 4, y + 4],
          [x + 1, y + 3], [x + 2, y + 3], [x + 3, y + 3], [x + 4, y + 3],
          [x + 1, y + 2], [x + 2, y + 2], [x + 3, y + 2], [x + 4, y + 2],
          [x + 1, y + 1], [x + 2, y + 1], [x + 3, y + 1], [x + 4, y + 1]]

    neighbor_8 = [[x + 5, y + 4], [x + 6, y + 4], [x + 7, y + 4], [x + 8, y + 4],
          [x + 5, y + 3], [x + 6, y + 3], [x + 7, y + 3], [x + 8, y + 3],
          [x + 5, y + 2], [x + 6, y + 2], [x + 7, y + 2], [x + 8, y + 2],
          [x + 5, y + 1], [x + 6, y + 1], [x + 7, y + 1], [x + 8, y + 1]]

    neighbor_9 = [[x - 8, y - 4], [x - 7, y - 4], [x - 6, y - 4], [x - 5, y - 4],
          [x - 8, y - 3], [x - 7, y - 3], [x - 6, y - 3], [x - 5, y - 3],
          [x - 8, y - 2], [x - 7, y - 2], [x - 6, y - 2], [x - 5, y - 2],
          [x - 8, y - 1], [x - 7, y - 1], [x - 6, y - 1], [x - 5, y - 1]]

    neighbor_10 = [[x - 4, y - 4], [x - 3, y - 4], [x - 2, y - 4], [x - 1, y - 4],
           [x - 4, y - 3], [x - 3, y - 3], [x - 2, y - 3], [x - 1, y - 3],
           [x - 4, y - 2], [x - 3, y - 2], [x - 2, y - 2], [x - 1, y - 2],
           [x - 4, y - 1], [x - 3, y - 1], [x - 2, y - 1], [x - 1, y - 1]]

    neighbor_11 = [[x + 1, y - 4], [x + 2, y - 4], [x + 3, y - 4], [x + 4, y - 4],
           [x + 1, y - 3], [x + 2, y - 3], [x + 3, y - 3], [x + 4, y - 3],
           [x + 1, y - 2], [x + 2, y - 2], [x + 3, y - 2], [x + 4, y - 2],
           [x + 1, y - 1], [x + 2, y - 1], [x + 3, y - 1], [x + 4, y - 1]]

    neighbor_12 = [[x + 5, y - 4], [x + 6, y - 4], [x + 7, y - 4], [x + 8, y - 4],
           [x + 5, y - 3], [x + 6, y - 3], [x + 7, y - 3], [x + 8, y - 3],
           [x + 5, y - 2], [x + 6, y - 2], [x + 7, y - 2], [x + 8, y - 2],
           [x + 5, y - 1], [x + 6, y - 1], [x + 7, y - 1], [x + 8, y - 1]]

    neighbor_13 = [[x - 8, y - 8], [x - 7, y - 8], [x - 6, y - 8], [x - 5, y - 8],
           [x - 8, y - 7], [x - 7, y - 7], [x - 6, y - 7], [x - 5, y - 7],
           [x - 8, y - 6], [x - 7, y - 6], [x - 6, y - 6], [x - 5, y - 6],
           [x - 8, y - 5], [x - 7, y - 5], [x - 6, y - 5], [x - 5, y - 5]]

    neighbor_14 = [[x - 4, y - 8], [x - 3, y - 8], [x - 2, y - 8], [x - 1, y - 8],
           [x - 4, y - 7], [x - 3, y - 7], [x - 2, y - 7], [x - 1, y - 7],
           [x - 4, y - 6], [x - 3, y - 6], [x - 2, y - 6], [x - 1, y - 6],
           [x - 4, y - 5], [x - 3, y - 5], [x - 2, y - 5], [x - 1, y - 5]]

    neighbor_15 = [[x + 1, y - 8], [x + 2, y - 8], [x + 3, y - 8], [x + 4, y - 8],
           [x + 1, y - 7], [x + 2, y - 7], [x + 3, y - 7], [x + 4, y - 7],
           [x + 1, y - 6], [x + 2, y - 6], [x + 3, y - 6], [x + 4, y - 6],
           [x + 1, y - 5], [x + 2, y - 5], [x + 3, y - 5], [x + 4, y - 5]]

    neighbor_16 = [[x + 5, y - 8], [x + 6, y - 8], [x + 7, y - 8], [x + 8, y - 8],
           [x + 5, y - 7], [x + 6, y - 7], [x + 7, y - 7], [x + 8, y - 7],
           [x + 5, y - 6], [x + 6, y - 6], [x + 7, y - 6], [x + 8, y - 6],
           [x + 5, y - 5], [x + 6, y - 5], [x + 7, y - 5], [x + 8, y - 5]]

    neighbors = neighbor_1+neighbor_2+neighbor_3+neighbor_4+neighbor_5+neighbor_6+neighbor_7+neighbor_8+ \
                neighbor_9+neighbor_10+neighbor_11+neighbor_12+neighbor_13+neighbor_14+neighbor_15+neighbor_16

    return neighbors

def gen_arc_histogram(direction, histogram):
    if direction > (2*math.pi)/3 and direction <= math.pi/3:
        histogram[0]+=1
    elif direction > math.pi/3 and direction <= math.pi/6:
        histogram[1]+=1
    elif direction > math.pi/6 and direction <= (11*math.pi)/6:
        histogram[2]+=1
    elif direction > (11*math.pi)/6 and direction <= (5*math.pi)/3:
        histogram[3]+=1
    elif direction > (5*math.pi)/3 and direction <= (4*math.pi)/3:
        histogram[4]+=1
    elif direction > (4*math.pi)/3 and direction <= (5*math.pi)/4:
        histogram[5]+=1
    elif direction > (5*math.pi)/4 and direction <= math.pi:
        histogram[6]+=1
    elif direction > math.pi and direction <= (2*math.pi)/3:
        histogram[7]+=1
    return histogram


if __name__ == '__main__':
    # test
    print(gen_neighbors(5, 4))