import pandas as pd
from ensemble_boxes import weighted_boxes_fusion

def run_ensemble():
    # 앙상블할 파일 지정
    file_names = [
        'vlk4jvob.csv',
        '91fz7ntu.csv',
        '8n81iwq8.csv',
        '13sdu82t.csv',
        '0qti3e4e.csv',
    ]
    
    # 데이터 로드
    dfs = [pd.read_csv(f) for f in file_names]
    image_ids = dfs[0]['image_id'].unique()
    
    # 이미지 해상도 설정
    IMG_WIDTH = 976.0 
    IMG_HEIGHT = 1280.0
    
    ensemble_results = []
    
    # 이미지별로 WBF 수행
    for img_id in image_ids:
        boxes_list, scores_list, labels_list = [], [], []
        
        for df in dfs:
            df_img = df[df['image_id'] == img_id]
            if len(df_img) == 0:
                continue
                
            b_list, s_list, l_list = [], [], []
            for _, row in df_img.iterrows():
                x_min = row['bbox_x'] / IMG_WIDTH
                y_min = row['bbox_y'] / IMG_HEIGHT
                x_max = (row['bbox_x'] + row['bbox_w']) / IMG_WIDTH
                y_max = (row['bbox_y'] + row['bbox_h']) / IMG_HEIGHT
                
                b_list.append([
                    max(0.0, min(1.0, x_min)), max(0.0, min(1.0, y_min)), 
                    max(0.0, min(1.0, x_max)), max(0.0, min(1.0, y_max))
                ])
                s_list.append(row['score'])
                l_list.append(int(row['category_id']))
                
            boxes_list.append(b_list)
            scores_list.append(s_list)
            labels_list.append(l_list)
            
        if not boxes_list: 
            continue
            
        # WBF 적용
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list, scores_list, labels_list,
            weights=[2.5, 2.0, 1.0, 1.5, 1.5], 
            iou_thr=0.5, 
            skip_box_thr=0.1
        )
        
        # 정규화 해제 및 포맷 원복
        for box, score, label in zip(boxes, scores, labels):
            x_min_n, y_min_n, x_max_n, y_max_n = box
            
            bbox_x = x_min_n * IMG_WIDTH
            bbox_y = y_min_n * IMG_HEIGHT
            bbox_w = (x_max_n - x_min_n) * IMG_WIDTH
            bbox_h = (y_max_n - y_min_n) * IMG_HEIGHT
            
            ensemble_results.append({
                'image_id': img_id,
                'category_id': int(label),
                'bbox_x': int(round(bbox_x)),
                'bbox_y': int(round(bbox_y)),
                'bbox_w': int(round(bbox_w)),
                'bbox_h': int(round(bbox_h)),
                'score': round(float(score), 5)
            })

    # DF 생성
    df_ensemble = pd.DataFrame(ensemble_results)
    
    # Top-4 필터링 (이미지당 점수가 높은 4개만 남김)
    df_ensemble = df_ensemble.sort_values(by=['image_id', 'score'], ascending=[True, False])
    df_ensemble = df_ensemble.groupby('image_id').head(4).reset_index(drop=True)
    
    # annotation_id 부여 및 컬럼 정렬
    df_ensemble['annotation_id'] = range(1, len(df_ensemble) + 1)
    cols = ['annotation_id', 'image_id', 'category_id', 'bbox_x', 'bbox_y', 'bbox_w', 'bbox_h', 'score']
    df_ensemble = df_ensemble[cols]
    
    # 최종 결과 저장
    output_name = 'submission.csv'
    df_ensemble.to_csv(output_name, index=False)
    print(f"[저장 완료] 총 BBox 개수: ({len(df_ensemble)})")

if __name__ == "__main__":
    run_ensemble()