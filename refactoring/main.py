# json 받아다가 데이터 전처리부터 KMO, EFA, CFA, 회귀분석까지 수행하는 코드
# 단, 편의성을 위해 각 단계마다 결과 데이터를 저장함



from util.packages import *
from util import a00_task_executor





def main() :

    # args = setup_parser().parse_args()
    # param = load_json(args.config)

    # for k, v in param.items():
    #     setattr(args, k, v)

    def dict_to_namespace(d):
        if isinstance(d, dict):
            return SimpleNamespace(**{
                k: dict_to_namespace(v) for k, v in d.items()
            })
        elif isinstance(d, list):
            return [dict_to_namespace(x) for x in d]
        else:
            return d

    args = setup_parser().parse_args()
    param = load_json(args.config)

    merged = {**vars(args), **param}
    args = dict_to_namespace(merged)
    


    # 랜덤시드는 글로벌로 지정
    RANDOM_SEED = args.RANDOM_SEED
    np.random.seed(RANDOM_SEED)

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
            a00_task_executor.preprocess(args.task_1, RANDOM_SEED, DATA_DIR, RESULTS_DIR)
        elif task == 2 :
            # KMO, Bartlett's test 함수
            a00_task_executor.kmo_bartlett(args.task_2)
        elif task == 3 :
            # EFA 함수
            a00_task_executor.efa(args.task_3, DATA_DIR, RESULTS_DIR)
        elif task == 4 :
            # CFA 함수
            a00_task_executor.cfa(args.task_4, RESULTS_DIR)
        elif task == 5 :
            # outcome test 함수 (예: 회귀분석)
            a00_task_executor.outcome_check(args.task_5, RESULTS_DIR, RANDOM_SEED)
        elif task == 6 : 
            # 번외 0 : 2011, 2018 데이터셋 outcome 변수만 추리도록 작업하기
            a00_task_executor.data_rework(args.task_6, RESULTS_DIR, RANDOM_SEED, DATA_DIR)
        elif task == 7 : 
            # 번외 1 : 정제 끝난 데이터로 임시 outcome test
            a00_task_executor.primal_var_check(args.task_7, RESULTS_DIR, RANDOM_SEED)
        elif task == 8 : 
            # factor score을 이용해 특정 outcome을 대상으로 regression 수행
            a00_task_executor.regression_test(args.task_8, RESULTS_DIR, RANDOM_SEED)
            





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
    