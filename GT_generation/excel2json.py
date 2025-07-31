import os
import pandas as pd
import json
import cv2

# --- 配置项 ---
clips_output_folder = './clips_vr' 
video_base_path = './datasets_vr'
excel_dir = './excel_vr'
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
fps = 30

# 长动作配置
long_actions = {
    "Moving using controller",
    "Catching fishes using net",
    "Waving hammer",
    "Waving laser",
    "Shooting"
}
clip_len = 150
stride = 180
video_format = 'mp4'
video_counter = 1


def action_details(act: str) -> str:
    map_act = {
        "Walking": "The person is walking forward. Their legs move alternately with relaxed arm swings.The background gently sways from side to side, suggesting curved or circular movement.",
        "Running": "The person is running quickly. Their arms pump back and forth while their feet hit the ground rapidly. As they move, The VR scene slightly shakes during the motion, enhancing the sense of movement.",
        "Jumping": "The person is jumping up. They bend their knees and push off the ground before landing softly. The VR scene moves briefly upward during the jump, then downward as they land, enhancing the sense of vertical motion.",
        "Bending down and up": "The person is bending down and up. They lower their upper body toward the ground and then return to an upright posture. The VR scene shifts slightly downward as they bend, and then moves upward as they rise, reflecting the change in head position.",
        "Standing": "The person is standing still. Their body remains upright and relaxed without major movement. The VR screen remains completely still, indicating a fully static moment in the virtual environment.",
        "Squatting": "The person is squatting down. They bend their knees and lower their hips close to the ground while maintaining balance. As they move downward, the VR perspective also shifts accordingly, simulating a natural change in viewpoint.",
        "Raising hand": "The person is raising one hand. Their arm lifts smoothly above shoulder level, with the rest of the body remaining still. The VR scene stays stable, as there is no significant change in head or body position.",
        "Shooting": "The person is shooting. They raise their arms and extend one finger to press the VR controller trigger, maintaining a steady forward posture during the action. While firing, the VR scene may stay still or move slightly depending on directional input, accompanied by continuous shot visuals.",
        "Waving hammer": "The person is waving a hammer. They swing one arm in wide arcs while holding a virtual hammer, using the shoulder and elbow to generate forceful side-to-side or overhead motions. As the hammer is waved, it visibly swings from side to side within the user’s field of view.",
        "Throwing hammer": "The person is throwing a hammer. They raise their arm and swing it forward in a forceful motion while pressing the VR controller, simulating the release of a hammer.A virtual hammer is launched forward in the VR environment, creating a strong visual cue.",
        "Waving laser": "The person is waving a laser. They lift one arm and swing it downward in a forceful motion, simulating the use of a handheld laser device.A visible laser beam appears and follows the arm’s path. The VR scene may flash or briefly vibrate to emphasize the power of the motion",
        "Cutting using laser": "The person is cutting with a laser tool. They move one hand slowly and steadily in a controlled slicing motion, simulating a precise laser cut.A focused laser beam appears along the cutting path. As the beam is active, incoming objects from the front are deflected or repelled upon contact.",
        "Moving using controller": "The person is moving using a VR controller. Their hands are performing subtle joystick movements or button presses to control their direction and pace.The VR scene shifts continuously in the chosen direction, simulating smooth locomotion. The viewpoint moves forward, backward, or sideways depending on input.",
        "Catching fishes using net": "The person is catching fish with a net. One arm performs a scooping motion forward and downward, mimicking the act of dipping a net into water. The VR environment simulates a water surface with fish visible. As the hand moves, virtual ripples form, and a catching animation plays when fish are captured.",
        "Grabbing and collecting box": "The person is grabbing and collecting a box. Their hands move carefully to lift and hold the item. The box becomes attached to or follows the player’s hands in the VR scene. The viewpoint may shift slightly if the player leans forward during the interaction, but the overall scene remains mostly stable.",
        "Measuring length": "The person is measuring length. They move one hand steadily through the virtual space, tracing a straight line horizontally or vertically to simulate a measurement action. A virtual ruler, laser guide, or measurement line appears along the hand’s path. The VR scene remains completely stable to support precision, with no major background or viewpoint movement. ",
        "Waving sword": "The person is waving a sword. They raise one arm and swing it forcefully through the air in wide arcs, simulating a sword-slashing motion. The sword visibly swings from side to side in the VR scene. As it moves, incoming objects are deflected or knocked back upon contact, providing immediate physical feedback.",
        "Bowling": "The person is bowling. They lean forward and swing one arm low and fast in a smooth underhand motion to simulate releasing a bowling ball. A white ball rolls forward along the virtual lane. The VR scene stays mostly stable, with slight camera movement for realism.",
        "Picking and Placing": "The person is picking up and placing objects.They bend slightly at the waist and use both hands to grab items with the VR controllers, then move and release them into target positions. Objects follow the hand movement smoothly, snapping into place when released. The VR scene remains mostly stable, with minor viewpoint shifts if the person leans."
    }

    if Act not in map_act:
        print(f"[❌] 未知动作 Act: '{Act}' 文件: {excel_path}, sheet: '{scene}', 行号: {row_idx + 2}")
    return map_act[act]

# --- 遍历 vr_01 ~ vr_15 + L/R/C ---
output_folder = './descriptions_vr'
os.makedirs(output_folder, exist_ok=True)
for i in range(1, 16):
    video_number = f"{i:02d}"
    video_csv_records = []
    video_json_records = []
    excel_path = os.path.join(excel_dir, f'DataCollection_{video_number}.xlsx')
    for video_direction in ['L', 'R', 'C']:
        if (video_number == "08" or video_number == "09" or video_number == "15") and video_direction == "R":
            print(f"[跳过] {video_number}{video_direction}")
            continue
        sheet_video_map = {
            'museum':   f'{video_number}_museum_{video_direction}',
            'bowling':  f'{video_number}_bowling_{video_direction}',
            'gallery':  f'{video_number}_gallery_{video_direction}',
            'travel':   f'{video_number}_travel_{video_direction}',
            'boss':     f'{video_number}_boss_{video_direction}',
            'candy':    f'{video_number}_candy_{video_direction}'
        }
        for scene, video_file in sheet_video_map.items():
            video_path = os.path.join(video_base_path, f"{video_file}.{video_format}")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print("Error: Could not open video.")
                exit()
        
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
            os.makedirs(output_folder, exist_ok=True)
            df = pd.read_excel(excel_path, sheet_name=scene)
            rep_start_cols = [col for col in df.columns if 'Repetition' in col and 'Start' in col]
            rep_end_cols = [col for col in df.columns if 'Repetition' in col and 'End' in col]
            rep_nums = set()
            for col in rep_start_cols + rep_end_cols:
                try:
                    rep_num = int(col.split()[1])
                    rep_nums.add(rep_num)
                except:
                    continue
            num_repetitions = max(rep_nums) if rep_nums else 0

            for row_idx, row in df.iterrows():
                for rep in range(1, num_repetitions + 1):
                    start_col = f"Repetition {rep} Start"
                    end_col = f"Repetition {rep} End"

                    start = row.get(start_col)
                    end = row.get(end_col)

                    if pd.notna(start) and pd.notna(end):
                        Act = str(df.iloc[row_idx, 0]).strip()
                        Dr = str(df.iloc[row_idx, 1]).strip()
                        scene_name = video_file.split('_')[1]
                        start_frame = int(start)
                        end_frame = int(end)

                        if Act in long_actions:
                            rep_id = 1
                            for clip_start in range(start_frame, end_frame - clip_len + 1, stride):
                                clip_end = clip_start + clip_len
                                clip_filename = os.path.join(clips_output_folder,f"{video_number}_{scene_name}_{video_direction}_{Act}_{Dr}_rep{rep_id}_{clip_start}_{clip_end}.{video_format}")
                                if not os.path.exists(clip_filename):
                                    out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))
                                    cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
                                    for i in range(clip_start, clip_end):
                                        ret, frame = cap.read()
                                        if not ret:
                                            break
                                        out.write(frame)
                                    out.release()
                                    print(f"Saved: {clip_filename}")
                                else:
                                    print(f"[SKIPPED] {clip_filename} already exists.")
                                video_id = f"{video_counter:04d}"
                                video_csv_records.append({
                                    "no.": video_id,
                                    "video_name": clip_filename,
                                    "action": Act,
                                })
                                video_json_records.append({
                                    "video": os.path.basename(clip_filename),
                                    "conversations": [
                                        {"from": "user", "value": "<video>What is the person doing? Focus on the human actions."},
                                        {"from": "assistant", "value": action_details(Act)}
                                    ]
                                })
                                video_counter += 1
                                rep_id += 1
                        else:
                            clip_filename = os.path.join(clips_output_folder,f"{video_number}_{scene_name}_{video_direction}_{Act}_{Dr}_rep{rep}_{start_frame}_{end_frame}.{video_format}")
                            if not os.path.exists(clip_filename):
                                out = cv2.VideoWriter(clip_filename, fourcc, fps, (width, height))
                                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                                for i in range(start_frame, end_frame + 1):
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    out.write(frame)
                                out.release()
                                print(f"Saved: {clip_filename}")
                            else:
                                print(f"[SKIPPED] {clip_filename} already exists.")
                            video_id = f"{video_counter:04d}"
                            video_csv_records.append({
                                "no.": video_id,
                                "video_name": clip_filename,
                                "action": Act,
                            })
                            video_json_records.append({
                                "video": os.path.basename(clip_filename),
                                "conversations": [
                                    {"from": "user", "value": "<video>What is the person doing? Focus on the human actions."},
                                    {"from": "assistant", "value": action_details(Act)}
                                ]
                            })
                            video_counter += 1

    # 保存当前编号 + 方向的 CSV 和 JSON
    csv_output_path = os.path.join(output_folder, f"{video_number}.csv")
    json_output_path = os.path.join(output_folder, f"{video_number}.json")

    pd.DataFrame(video_csv_records).to_csv(csv_output_path, index=False, encoding="utf-8-sig")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(video_json_records, f, indent=2, ensure_ascii=False)

    print(f"生成完成：{json_output_path}（{len(video_json_records)} 条）")