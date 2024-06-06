import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf"  # 시스템에 설치된 폰트 경로로 변경
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 마이너스 기호 깨짐 문제 해결
plt.rcParams['axes.unicode_minus'] = False

#============================================================================================================================
# 데이터 파일 로드
file_path = 'simulation_results_with_parameters_test.csv'
df = pd.read_csv(file_path)

# 열 이름의 공백 제거
df.columns = df.columns.str.strip()

# 정확한 열 이름 설정
exact_column_name = 'Average Escape Time (s)'

# 'Combination Number'로 그룹화하여 평균 탈출 시간 계산
grouped = df.groupby('Combination Number')[exact_column_name].mean().reset_index()

# 평균 탈출 시간을 기준으로 조합 번호 정렬
sorted_combinations = grouped.sort_values(by=exact_column_name)['Combination Number']

# 정렬된 조합 번호를 기준으로 원래 데이터프레임 재정렬
df['Combination Number'] = pd.Categorical(df['Combination Number'], categories=sorted_combinations, ordered=True)
df_sorted = df.sort_values('Combination Number')

# 점수 계산
min_time = grouped[exact_column_name].min()
max_time = grouped[exact_column_name].max()
grouped['Score'] = 100 * (1 - (grouped[exact_column_name] - min_time) / (max_time - min_time))

# 점수를 기준으로 정렬된 데이터프레임 생성
sorted_scores = grouped.sort_values(by='Score', ascending=False)

# 점수를 기준으로 정렬된 데이터프레임을 CSV 파일로 저장
sorted_scores.to_csv('sorted_scores.csv', index=False)

print(f'Sorted scores saved to: sorted_scores.csv')

# 상위 20개와 하위 20개 조합 선택
top_20 = sorted_scores.head(20)
bottom_20 = sorted_scores.tail(20)

# 상위 20개와 하위 20개 조합을 각각 저장하여 두 열로 나눔
combined_scores = pd.DataFrame({
    'Top Combination': top_20['Combination Number'].values,
    'Top Average Time (s)': top_20[exact_column_name].values,
    'Top Score': top_20['Score'].values,
    'Bottom Combination': bottom_20['Combination Number'].values[::-1],
    'Bottom Average Time (s)': bottom_20[exact_column_name].values[::-1],
    'Bottom Score': bottom_20['Score'].values[::-1]
})

# 소수점 자릿수를 줄이기
combined_scores = combined_scores.round({'Top Average Time (s)': 2, 'Top Score': 2, 'Bottom Average Time (s)': 2, 'Bottom Score': 2})

# subplot을 사용하여 모든 그래프를 2x2 레이아웃으로 그리기
fig, axs = plt.subplots(2, 2, figsize=(15, 15))  # 각 플롯의 크기를 적절히 줄임

# 각 조합 번호에 대한 평균 탈출 시간 분포를 나타내는 박스 플롯 그리기
sns.boxplot(x='Combination Number', y=exact_column_name, data=df_sorted, ax=axs[0, 0])
axs[0, 0].set_title('각 조합 번호에 대한 평균 탈출 시간 분포 (오름차순 정렬)')
axs[0, 0].set_xlabel('조합 번호')
axs[0, 0].set_ylabel('평균 탈출 시간 (초)')
axs[0, 0].tick_params(axis='x', rotation=90)

# 상위 20개 및 하위 20개 조합 점수 표 그리기
axs[0, 1].axis('tight')
axs[0, 1].axis('off')
table = axs[0, 1].table(cellText=combined_scores.values, colLabels=combined_scores.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.2, 1.2)  # 표의 스케일을 조정
axs[0, 1].set_title('상위 20개 및 하위 20개 조합 점수')

# 평균 탈출 시간 순위 막대 그래프
sorted_grouped = grouped.sort_values(by='Average Escape Time (s)')
sns.barplot(x='Combination Number', y='Average Escape Time (s)', data=sorted_grouped, palette='coolwarm', order=sorted_grouped['Combination Number'], ax=axs[1, 0])
axs[1, 0].set_title('평균 탈출 시간 순위 막대 그래프')
axs[1, 0].set_xlabel('조합 번호')
axs[1, 0].set_ylabel('평균 탈출 시간 (초)')
axs[1, 0].tick_params(axis='x', rotation=90)

# 남은 공간은 빈 곳으로 남기기
axs[1, 1].axis('off')

plt.tight_layout()

# 플롯을 파일로 저장
combined_plot_path = 'combined_plots_2x2.png'
plt.savefig(combined_plot_path)

# 플롯 표시
#plt.show()

print(f'Combined plot saved to: {combined_plot_path}')

#============================================================================================================================
# 창문 그리기 함수
def draw_window_configuration(ax, configuration_number):
    config = df.iloc[configuration_number - 1].to_dict()
    print(f"Configuration {configuration_number}: {config}")  # 선택된 구성 출력

    ax.set_xlim(0, room_Z)
    ax.set_ylim(0, room_X)

    # 방의 경계 그리기
    rect = plt.Rectangle((0, 0), room_Z, room_X, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)

    # 마지막 조합을 회색 박스로 먼저 그리기
    for side, coords in last_config.items():
        for coord in coords:
            #print(coord[0])
            if side == 'left':
                rect = plt.Rectangle((0, coord[0]), 0.1, window_length, color='grey')
            elif side == 'top':
                rect = plt.Rectangle((coord[0], room_X - 0.1), window_length, 0.1, color='grey')
            elif side == 'right':
                rect = plt.Rectangle((room_Z - 0.1, room_X - coord[1] - window_length), 0.1, window_length, color='grey')
            elif side == 'bottom':
                rect = plt.Rectangle((coord[0], 0), window_length, 0.1, color='grey')
            ax.add_patch(rect)

    # 선택된 조합을 빨간색 박스로 그리기
    for side, coords in config.items():
        for coord in coords:
            print(coord)
            if side == 'left':
                rect = plt.Rectangle((0, coord[0]), 0.1, window_length, color='red')
            elif side == 'top':
                rect = plt.Rectangle((coord[0], room_X - 0.1), window_length, 0.1, color='red')
            elif side == 'right':
                rect = plt.Rectangle((room_Z - 0.1, room_X - coord[1] - window_length), 0.1, window_length, color='red')
            elif side == 'bottom':
                rect = plt.Rectangle((coord[0], 0), window_length, 0.1, color='red')
            ax.add_patch(rect)

    ax.set_aspect('equal', adjustable='box')

# 좌표를 튜플 형태로 변환
def parse_coordinates(coord_str):
    try:
        coords = eval(coord_str)
        if isinstance(coords, tuple):
            return [coords]
        elif isinstance(coords, list):
            return [tuple(c) for c in coords]
        else:
            return []
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        return []

# 파일에서 room_Z와 room_X 값 읽기
file_path = 'C:\\Users\\badag\\PJ_Algorithm\\base_model\\input_Lidar.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 필요한 값 추출
for line in lines:
    if 'room_Z' in line:
        room_Z = float(line.split('=')[1].strip())
    elif 'room_X' in line:
        room_X = float(line.split('=')[1].strip())

# CSV 파일에서 창문 조합 읽기
df = pd.read_csv('window_configurations.csv')

for column in ['left', 'top', 'right', 'bottom']:
    df[column] = df[column].apply(parse_coordinates)

print(df.head())  # 변환된 좌표를 확인하기 위해 데이터프레임 출력

# 창문 크기 설정
window_length = 1

# 마지막 조합 저장
last_config = df.iloc[-1].to_dict()

# 상위 4개의 조합 번호와 하위 4개의 조합 번호 가져오기
top_4_combinations = top_20['Combination Number'].values[:4]
bottom_4_combinations = bottom_20['Combination Number'].values[-4:]

# 상위 4개와 하위 4개 조합의 창문을 그릴 subplot 생성
fig, axs = plt.subplots(2, 4, figsize=(20, 10))

# 상위 4개의 조합 창문 그리기
for i, combination in enumerate(top_4_combinations):
    draw_window_configuration(axs[0, i], combination)
    axs[0, i].set_title(f'Top {i+1} Combination')

# 하위 4개의 조합 창문 그리기
for i, combination in enumerate(bottom_4_combinations):
    draw_window_configuration(axs[1, i], combination)
    axs[1, i].set_title(f'Bottom {i+1} Combination')

plt.tight_layout() # 서브플롯(subplot) 간의 간격 조정

# 플롯을 파일로 저장
window_plot_path = 'top_bottom_4_combinations.png'
plt.savefig(window_plot_path)

print(f'Window plot saved to: {window_plot_path}')

plt.show()