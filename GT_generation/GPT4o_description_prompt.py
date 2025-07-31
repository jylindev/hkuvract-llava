import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import json
import os
import ast
import random
import shutil
from decord import VideoReader, cpu
import numpy as np
from multiprocessing import Pool, Manager
import openai
from openai import OpenAI
from datetime import datetime
from tqdm import tqdm
import time


API_KEY = "sk-XXX"

client = OpenAI()

# Define constants
DATASET_NAME = "default"
# SAVE_FRAME_SAVE_PATH_ROOT = "./images"
# if not os.path.exists(SAVE_FRAME_SAVE_PATH_ROOT):
#     os.makedirs(SAVE_FRAME_SAVE_PATH_ROOT)

SAVE_DESCRIPTION_PATH = "./descriptions"
if not os.path.exists(SAVE_DESCRIPTION_PATH):
    os.makedirs(SAVE_DESCRIPTION_PATH)


video_progression_structures = [
    "The video begins with..., develops through..., and wraps up with...",
    "Initially, the video shows..., followed by..., and finally...",
    "The video starts with..., transitions to..., and ends with...",
    "In the beginning, the video presents..., then..., and it finishes with...",
    "The video sequence commences with..., moves on to..., and culminates in...",
    "First, the video displays..., then it illustrates..., and lastly, it concludes with...",
    "The video starts by..., continues with..., and ends by...",
    "The video opens with..., followed by..., and concludes with...",
    "The video kicks off with..., progresses through..., and finishes with...",
    "At the outset, the video shows..., which leads to..., and ultimately ends with...",
    "The narrative of the video begins with..., evolves into..., and resolves with...",
    "Starting with..., the video then moves to..., and finally...",
    "The video sequence begins with..., develops into..., and concludes by...",
    "First, we see in the video..., next..., and finally...",
    "The video initiates with..., continues by..., and concludes with...",
    "Initially, the video presents..., followed by..., and ending with...",
    "The video commences with..., advances by..., and finishes with...",
    "The video opens with..., transitions into..., and wraps up with...",
    "In the beginning, the video shows..., then it shifts to..., and ultimately...",
    "The video starts with..., moves through..., and ends at...",
    "The video begins with..., progresses by..., and concludes with..."
]

# summarize_structure= "The clip begins with..., progresses by..., and concludes with..."

# special_video_summarize_critria = '\n'

first_clip_system = '''
### Task:
You are an expert in understanding scene transitions based on visual features in a video. You are requested to create the descriptions for the current clip sent to you,  which includes multiple sequential frames.

#### Guidelines For Clip Description:
- Analyze the narrative progression implied by the sequence of frames, interpreting the sequence as a whole.
- Note that since these frames are extracted from a clip, adjacent frames may show minimal differences. These should not be interpreted as special effects in the clip.
- If text appears in the frames, you must describe the text in its original language and provide an English translation in parentheses. For example: 书本 (book). Additionally, explain the meaning of the text within its context.
- When referring to people, use their characteristics, such as clothing, to distinguish different people.
- **IMPORTANT** Please provide as many details as possible in your description, including colors, shapes, and textures of objects, actions and characteristics of humans, as well as scenes and backgrounds. 


### Output Format:
1. Your output should be formed in a JSON file.
2. Only provide the Python dictionary string.

Your response should look like this: {"Clip Level Description": "The clip begins with..., progresses by..., and concludes with..."}
'''

first_clip ='''
Please give me the description of the current clip.
'''

other_prompt_system = '''
### Task:
You are an expert in understanding scene transitions based on visual features in a video. There is a video including multiple sequential clips (clip-1,clip-2,XXX). Given the description for these clips (clip-1,clip-2,...,) as the context, you are requested to create the descriptions for the current clip sent to you,  which includes multiple sequential frames.

#### Guidelines For Clip Description:
- Your description should see the description of previous clips as context.
- Analyze the narrative progression implied by the sequence of frames, interpreting the sequence as a whole.
- Note that since these frames are extracted from a clip, adjacent frames may show minimal differences. These should not be interpreted as special effects in the clip.
- Note that some objects and scenes shown in the previous clips might not shown in the current clip. Be carefully do not assume the same object and scenes shown in every clips.
- If text appears in the frames, you must describe the text in its original language and provide an English translation in parentheses. For example: 书本 (book). Additionally, explain the meaning of the text within its context.
- When referring to people, use their characteristics, such as clothing, to distinguish different people.
- **IMPORTANT** Please provide as many details as possible in your description, including colors, shapes, and textures of objects, actions and characteristics of humans, as well as scenes and backgrounds. 


### Output Format:
1. Your output should be formed in a JSON file.
2. Only provide the Python dictionary string.

Your response should look like this: {"Clip Level Description": "The clip begins with..., progresses by..., and concludes with..."}
'''

other_prompt = '''
### Description of Previous Clips (sorted in chronological order by number):
{previous_clip_description}

Please give me the description of the current clip.
'''

summarize_prompt_system = '''
### Task:
You are an expert at understanding clip descriptions in a video that includes {number_of_clip} clips. You are requested to create a video description by summarizing these clip descriptions chronologically.

#### Guidelines For Video Description:
- Since the clip descriptions are provided in chronological order, ensure that the video description is coherent and follows the same sequence. Avoid referring to the first or final frame of each clip as the first or final frame of the entire video.
- Include any text that appears in the clip, provide its English translation in parentheses, and explain the significance of each text within its context.
- The tone of the video description should be as if you are describing a video directly instead of summarizing the information from several clip descriptions. Therefore, avoid phrases found in the referred clip descriptions such as "The clip begins...", "As the clip progresses...", "The clip concludes", "The final/first frame", "The second clip begins with", "The final frames of this segment", etc
- **IMPORTANT** Include all details from the given clip descriptions in the video description. Try to understand of the theme of the video and provide a coherent narrative that connects all the clips together.

### Output Format:
1. Your output should be formed in a JSON file.
2. Only provide the Python dictionary string.
3. You can use various descriptive sentence structures to outline the narrative progression. One example is: {current_sentence_structure}

Your response should look like this: {{"Video Level Description": "YOUR DESCRIPTION HERE."}}
'''

summarize_prompt = '''
#### Clip Description (sorted in chronological order by number):
{all_clip_description}

Please give me the description of the video given the clip descriptions.
'''


# Function to load video and encode frames in base64
def load_video_base64(path):
    video = VideoReader(path, ctx=cpu(0), num_threads=1)
    video_id = os.path.basename(path).split(".")[0]
    cur_save_frame_save_path_root = f"{SAVE_FRAME_SAVE_PATH_ROOT}/{video_id}"
    if not os.path.exists(cur_save_frame_save_path_root):
        os.makedirs(cur_save_frame_save_path_root)

    # Save the video 
    # shutil.copy(path, f"{cur_save_frame_save_path_root}/{os.path.basename(path)}")

    # Get the original video's fps
    original_fps = round(video.get_avg_fps())

    base64Frames = []
    frame_counter = 0
    MAX_SIZE = 4 * 1024 * 1024  # 4MB in bytes
    for i, frame in enumerate(video):
        if i % original_fps == 0:
            frame_bgr = cv2.cvtColor(frame.asnumpy(), cv2.COLOR_RGB2BGR)  # Convert BGR to RGB
            _, buffer = cv2.imencode(".png", frame_bgr)
            buffer = base64.b64encode(buffer).decode("utf-8")
            while len(buffer.encode('utf-8')) > MAX_SIZE:  # Calculate the size of buffer in bytes
                # Reduce the image size by an increment of 10%
                width = int(frame_bgr.shape[1] * 0.9)
                height = int(frame_bgr.shape[0] * 0.9)
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation = cv2.INTER_AREA)
                _, buffer = cv2.imencode(".png", frame_bgr)
                buffer = base64.b64encode(buffer).decode("utf-8")
            base64Frames.append(buffer)
            frame_counter += 1

    return video_id, base64Frames

# Function to handle OpenAI Chat Completions with error handling
def testVisionOpenaiChatCompletions(base64Frames, query_system, query, version='gpt4v'):
    retries = 5
    for _ in range(retries):
        try:
            completion = client.chat.completions.create(
                model=version, 
                temperature=0,
                top_p=0.1,
                messages=[
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": [                        
                        {"type": "text", "text": query},
                        *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{x}"}}, base64Frames),
                    ]}
                ]
            )
            return completion
        except openai.RateLimitError as e:
            # print(f'ERROR: {e}')
            time.sleep(60)
            # import pdb; pdb.set_trace()
        except openai.OpenAIError as e:
            print(f'ERROR: {e}')
            return f"Unsuccessful: {e.message}"
    
    print(f"Failed after multiple retries.")
    return f"Unsuccessful: Failed after multiple retries."

def testOpenaiChatCompletions(query_system, query, version='gpt4'):
    retries = 5
    for _ in range(retries):
        try:
            completion = client.chat.completions.create(
                model=version, 
                temperature=0,
                top_p=0.1,
                messages=[
                    {"role": "system", "content": query_system},
                    {"role": "user", "content": query}
                ]
            )
            return completion
        # except Exception as e:
        #     print(f'ERROR: {e}')
        #     import pdb; pdb.set_trace()
        except openai.RateLimitError as e:
            # time.sleep(60)
            # print(f'ERROR: {e}')
            time.sleep(60)
        except openai.OpenAIError as e:
            print(f'ERROR: {e}')
            return f"Unsuccessful: {e.message}"
    
    print(f"Failed after multiple retries.")
    return f"Unsuccessful: Failed after multiple retries."

# Function to save description JSON to file
def save_description_json(video_id, ret, sum_ret,completion_tokens, prompt_tokens, total_tokens, prompts_system, prompts, day, version, all_success=False):
    filename = f"{SAVE_DESCRIPTION_PATH}/{DATASET_NAME}_{version}_{video_id}.json"
    with open(filename, "w", encoding='utf-8') as f:
        if all_success:
            json.dump({'clip_description': ret,'summary_description': sum_ret[:-1],'video_description':sum_ret[-1], 'completion_tokens': completion_tokens, 'prompt_tokens': prompt_tokens, 'total_tokens': total_tokens, 'prompts_system': prompts_system, 'prompts': prompts}, f, ensure_ascii=False)
        else:
            json.dump({'clip_description': ret,'summary_description': sum_ret, 'completion_tokens': completion_tokens, 'prompt_tokens': prompt_tokens, 'total_tokens': total_tokens, 'prompts_system': prompts_system, 'prompts': prompts}, f, ensure_ascii=False)
    print(f"Saved description for {video_id} to {filename}")


def save_error_json(video_id, ret, day, version,response):
    filename = f"{SAVE_DESCRIPTION_PATH}/{DATASET_NAME}_{version}_error.jsonl"
    with open(filename, "a+", encoding='utf-8') as f:
        f.write(f"{video_id}:{response}\n")
    print(f"Saved unsuccessful tag for {video_id} to {filename}")

def prompt_video_description(ret,last_time_summarize=False):
    query_system = summarize_prompt_system.format(number_of_clip=len(ret),current_sentence_structure=random.choice(video_progression_structures))     
    # import pdb; pdb.set_trace()   
    context = "\n".join(f"{idx + 1}.{clip}" for idx, clip in enumerate(ret))
    query = summarize_prompt.format(all_clip_description=context)
    response = testOpenaiChatCompletions(query_system, query, version)

    try:
        # new_text = ast.literal_eval(response.choices[0].message.content.replace('```json\n', ' ').replace('\n```', ' ').replace('\n','####'))["Video Level Description"]
        new_text = ast.literal_eval('{"'+response.choices[0].message.content.split("{")[1].split('"',1)[1].replace("\n\n","####").replace('```json\n', ' ').replace('\n```', ' '))["Video Level Description"]
    except:
        import pdb; pdb.set_trace()
        # new_text = response.choices[0].message.content.replace('```json\n', '').replace('\n```', '').replace('\n','####')
        new_text = '{"'+response.choices[0].message.content.split("{")[1].split('"',1)[1].replace("\n\n","####").replace('```json\n', ' ').replace('\n```', ' ')

    new_text = new_text.replace('####', '\n\n')

    if not last_time_summarize:
        new_text = new_text.replace("video","clip")
    
    # import pdb; pdb.set_trace()

    return new_text,response,query_system,query

# Main function to process a video
def process_video(version, path, day):
    ret = []
    sum_ret = []
    completion_tokens = []
    prompt_tokens = []
    total_tokens = []
    prompts_system = []
    prompts = []
    try:
        video_id, all_base64Frames = load_video_base64(path)
        if os.path.exists(f"{SAVE_DESCRIPTION_PATH}/{DATASET_NAME}_{version}_{video_id}.json"):
            cur_save_json = json.load(open(f"{SAVE_DESCRIPTION_PATH}/{DATASET_NAME}_{version}_{video_id}.json"))
            if "video_description" in cur_save_json:
                print(f"Already processed {video_id} for video description")
                return
            else:
                ret = cur_save_json["clip_description"]
                sum_ret = cur_save_json["summary_description"]
                completion_tokens = cur_save_json["completion_tokens"]
                prompt_tokens = cur_save_json["prompt_tokens"]
                total_tokens = cur_save_json["total_tokens"]
                promtps_system = cur_save_json["prompts_system"]
                prompts = cur_save_json["prompts"]
                print(f"Already processed {video_id} for {len(ret)} clip description")

        start_time = len(ret) * 10
        for i in range(start_time, len(all_base64Frames), 10):
            cur_start_time = i
            cur_end_time = min(i+10, len(all_base64Frames))
            base64Frames = all_base64Frames[i:i+10]
            if len(ret) == 0:
                query_system = first_clip_system
                query = first_clip
            else:
                # import pdb; pdb.set_trace()
                query_system = other_prompt_system
                # if len(sum_ret) != 0:
                if cur_start_time % 30 == 0:
                    pre_clip_desc = []
                else:
                    pre_clip_desc = ret[len(sum_ret)*3:cur_start_time//10]
                pre_desc = [sum_ret[-1]] + pre_clip_desc if len(sum_ret) != 0 else pre_clip_desc
                # import pdb; pdb.set_trace()
                context = "\n".join(f"{idx + 1}.{clip}" for idx, clip in enumerate(pre_desc))
                query = other_prompt.format(previous_clip_description=context)

            # import pdb; pdb.set_trace()
            response = testVisionOpenaiChatCompletions(base64Frames, query_system, query, version) 
            if isinstance(response,str):
                save_description_json(video_id, ret, sum_ret, completion_tokens, prompt_tokens, total_tokens, prompts_system, prompts, day, version)
                response = f"{response}; after the {len(ret)}th clip"
                save_error_json(video_id, ret, day, version, response)
                return

            try:  
                # new_text = ast.literal_eval(response.choices[0].message.content.replace('```json\n', ' ').replace('\n```', ' ').replace('\n','####'))["Clip Level Description"]
                new_text = ast.literal_eval('{"'+response.choices[0].message.content.split("{")[1].split('"',1)[1].replace("\n\n","####").replace('```json\n', ' ').replace('\n```', ' '))["Clip Level Description"]
            except:
                import pdb; pdb.set_trace()
                new_text = '{"'+response.choices[0].message.content.split("{")[1].split('"',1)[1].replace("\n\n","####").replace('```json\n', ' ').replace('\n```', ' ')
            
            new_text = new_text.replace('####', '\n\n')
            ret.append(new_text)
            completion_tokens.append(response.usage.completion_tokens)
            prompt_tokens.append(response.usage.prompt_tokens)
            total_tokens.append(response.usage.total_tokens)
            prompts_system.append(query_system)
            prompts.append(query)

            if cur_end_time % 30 == 0:
                if cur_end_time == len(all_base64Frames):
                    last_time_summarize = True
                else:
                    last_time_summarize = False
                clip_desc = ret[len(sum_ret)*3:]
                cur_desc = [sum_ret[-1]] + clip_desc if len(sum_ret) != 0 else clip_desc
                # import pdb; pdb.set_trace()
                new_text,response,query_system,query = prompt_video_description(cur_desc,last_time_summarize)
                sum_ret.append(new_text)
                completion_tokens.append(response.usage.completion_tokens)
                prompt_tokens.append(response.usage.prompt_tokens)
                total_tokens.append(response.usage.total_tokens)
                prompts_system.append(query_system)
                prompts.append(query)
        
        # import pdb; pdb.set_trace()
        if cur_end_time % 30 != 0:
            clip_desc = ret[len(sum_ret)*3:]
            cur_desc = [sum_ret[-1]] + clip_desc if len(sum_ret) != 0 else clip_desc
            new_text,response,query_system,query = prompt_video_description(cur_desc,True)
            sum_ret.append(new_text)
            completion_tokens.append(response.usage.completion_tokens)
            prompt_tokens.append(response.usage.prompt_tokens)
            total_tokens.append(response.usage.total_tokens)
            prompts_system.append(query_system)
            prompts.append(query)
                

    except Exception as e:
        print(f'ERROR: {e}')
    else:
        save_description_json(video_id, ret, sum_ret, completion_tokens, prompt_tokens, total_tokens, prompts_system, prompts, day, version, all_success=True)

# Main script
if __name__ == '__main__':
    debug = False
    day = datetime.now().strftime('%m%d')
    # version = 'gpt-4-1106-vision-preview'
    version = 'gpt-4o-2024-05-13'
    # version = 'gpt-4-turbo-2024-04-09'

    try:
        meta = f"meta.json"
        with open(meta) as f:
            data = json.load(f)

        cur_videos_list = data
        random.seed(0)
        random.shuffle(cur_videos_list)
        # cur_videos_list = cur_videos_list[:3]
    except FileNotFoundError:
        print("No video_duration.json file found, please check the path.")
        sys.exit(1)

    sucess_videos = set()
    for cur_json in os.listdir(SAVE_DESCRIPTION_PATH):
        sucess_videos.add(cur_json.split(".")[0])
        

    pool = Pool(processes=1)  # Adjust the number of processes as needed
    system_start_time = datetime.now()
    for cur_video in cur_videos_list[:]:
        # cur_video = "tiktok.mp4"
        video_id = os.path.basename(cur_video).split(".")[0]
        save_name = f"{DATASET_NAME}_{version}_{video_id}"
        # if save_name in sucess_videos:
        #     continue
        # process_video(version, cur_video, day)
        pool.apply_async(process_video, args=(version, cur_video, day))

    pool.close()
    pool.join()
    print(f"Total time taken: {datetime.now() - system_start_time}")
