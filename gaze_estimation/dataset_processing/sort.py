def process_line(line):
    line = line.strip()
    if '-' in line:
        start, end = map(int, line.split('-'))
        numbers = [str(num) for num in range(start, end + 1)]
        return '\n'.join(numbers)
    else:
        return line

# 입력 파일 경로
input_file_path = '/home/kaai/catkin_cw/src/cw/data_processing/right.txt'
# 출력 파일 경로
output_file_path = '/home/kaai/catkin_cw/src/cw/data_processing/right_out.txt'

# 결과 저장 변수
result = []

# 파일 읽기
with open(input_file_path, 'r') as input_file:
    # 각 라인 처리
    for line in input_file:
        processed_line = process_line(line)
        result.append(processed_line)

# 결과 저장
with open(output_file_path, 'w') as output_file:
    output_file.write('\n'.join(result))