import json

# 读取tracking结果（YOLO+DeepSORT的）
with open('tracking_results.json', 'r', encoding='utf-8') as f:
    tracking_data = json.load(f)

# 读取timesformer结果（行为识别的）
with open('../outputs.json', 'r', encoding='utf-8') as f:
    action_data = json.load(f)

with open("../Models/tracking.seq", "r", encoding='utf-8') as f:
    lines = f.readlines()

person_positions = {}
for line in lines:
    frame_id, track_id, x1, y1, x2, y2 = map(int, line.strip().split(","))
    if frame_id not in person_positions:
        person_positions[frame_id] = []
    person_positions[frame_id].append({
        "track_id": track_id,
        "bbox": [x1, y1, x2, y2]
    })


price_dict = {
    "AD Calcium Milk": 5,
    "Coca-Cola": 4,
    "Daliyuan Soft Bread": 3,
    "Kiss Burn Braised Beef Flavor": 2,
    "Kiss Burn Spicy Chicken Flavor": 2,
    "Lai Yi Tong Instant Noodles": 4,
    "RIO Lychee Flavor": 6,
    "RIO Strawberry Flavor": 6,
    "Tea Pi": 3,
    "Want Want Senbei": 2
}

def bboxes_intersect(boxA, boxB):
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

frame_gap_threshold = 45
recent_frame_tracker = {}  # person_id -> {item_name: last_frame}

final_result = {}

# 遍历每个人物
for track_id_str, actions in action_data.items():
    person_id = int(track_id_str)
    sorted_actions = sorted(actions, key=lambda x: x['start_frame'])

    stack = []
    for action in sorted_actions:
        act, start, end = action['action'], action['start_frame'], action['end_frame']
        if act in [0, 3]:
            stack.append((start, end))
        elif act == 2 and stack:
            stack.pop()

    hold_periods = stack
    if not hold_periods:
        continue

    for frame in tracking_data:
        frame_id = frame['frame_id']
        if not any(start <= frame_id <= end for (start, end) in hold_periods):
            continue

        people = person_positions.get(frame_id, [])
        person_bbox = None
        for p in people:
            if p['track_id'] == person_id:
                person_bbox = p['bbox']
                break
        if person_bbox is None:
            continue

        for item in frame['items']:
            item_name = item['item_name']
            item_bbox = item['bbox']
            if not bboxes_intersect(person_bbox, item_bbox):
                continue

            # 帧间隔限制判断
            if person_id not in recent_frame_tracker:
                recent_frame_tracker[person_id] = {}
            last_frame = recent_frame_tracker[person_id].get(item_name, -999)
            if frame_id - last_frame < frame_gap_threshold:
                continue  # 相隔帧太短，跳过

            # 更新记录
            recent_frame_tracker[person_id][item_name] = frame_id

            person_key = str(person_id)
            if person_key not in final_result:
                final_result[person_key] = {"items": {}, "total_price": 0}
            if item_name not in final_result[person_key]["items"]:
                final_result[person_key]["items"][item_name] = 1
            else:
                final_result[person_key]["items"][item_name] += 1
            final_result[person_key]["total_price"] += price_dict.get(item_name, 0)


with open('../final_shopping_result.json', 'w', encoding='utf-8') as f:
    json.dump(final_result, f, indent=4, ensure_ascii=False)

print("融合结果已保存至 final_shopping_result.json")