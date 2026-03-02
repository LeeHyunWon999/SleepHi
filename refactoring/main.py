# json 받아다가 데이터 전처리부터 KMO, EFA, CFA, 회귀분석까지 수행하는 코드
# 단, 편의성을 위해 각 단계마다 결과 데이터를 저장함




from util.packages import *
from util import a00_task_executor





def main() :

    args = setup_parser().parse_args()
    param = load_json(args.config)

    for k, v in param.items():
        setattr(args, k, v)

    # 랜덤시드는 글로벌로 지정
    RANDOM_SEED = args.random_seed
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    # 데이터 입력 / 결과 출력 경로 지정 (얘네도 글로벌)
    DATA_DIR = args.DATA_DIR
    RESULTS_DIR = args.RESULTS_DIR

    
    # args의 task_list 순서 목록 보고 순서대로 함수 실행
    for task in args.task_list :

        if task == 0 : 
            # 초반 기타 작업 함수
            a00_task_executor.etc_jobs(args.task_0)
        elif task == 1 :
            # 데이터 전처리 함수
            df_EFA, df_CFA, df_EFA_raw, df_CFA_raw = a00_task_executor.preprocess(args.task_1)
        elif task == 2 :
            # KMO, Bartlett's test 함수
            a00_task_executor.kmo_bartlett(args.task_2)
        elif task == 3 :
            # EFA 함수
            a00_task_executor.efa(args.task_3)
        elif task == 4 :
            # CFA 함수
            a00_task_executor.cfa(args.task_4)
        elif task == 5 :
            # outcome test 함수 (예: 회귀분석)
            a00_task_executor.outcome_test(args.task_5)
            





def load_json(path):
    with open(path) as data_file:
        param = json.load(data_file)

    return param

def setup_parser():
    parser   = argparse.ArgumentParser(description='SNN encoding')
    
    parser.add_argument('config')
   
    return parser


if __name__ =="__main__":

    main()
    