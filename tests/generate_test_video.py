import os
import cv2
import numpy as np
import math
import random

def create_synthetic_cctv_video(output_path: str = "data/sample_cctv.mp4", duration_sec: int = 15, fps: int = 25):
    """
    Generates a realistic synthetic CCTV video of a railway platform with:
    - Platform platform edge, track zone, concourse
    - Passenger figures walking with varying speeds
    - Simulated events: crowd surge, passenger fall, restricted track intrusion, panic scattering.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    
    # Pre-generate passengers
    # Each agent: [x, y, vx, vy, width, height, color_shirt, color_pants, is_fallen, fall_start_frame, has_fallen]
    num_agents = 22
    agents = []
    
    for i in range(num_agents):
        x = random.uniform(150, 700)
        y = random.uniform(220, 620)
        vx = random.uniform(-1.5, 1.5)
        vy = random.uniform(-0.8, 0.8)
        w = random.randint(38, 52)
        h = int(w * random.uniform(2.3, 2.8)) # Standing aspect ratio ~ 2.5
        shirt_color = (random.randint(40, 220), random.randint(40, 220), random.randint(40, 220))
        pants_color = (random.randint(20, 80), random.randint(20, 80), random.randint(20, 80))
        skin_color = (160, 195, 230)
        agents.append({
            'id': i,
            'x': x, 'y': y,
            'vx': vx, 'vy': vy,
            'w': w, 'h': h,
            'target_w': w, 'target_h': h,
            'shirt_color': shirt_color,
            'pants_color': pants_color,
            'skin_color': skin_color,
            'state': 'normal', # 'normal', 'falling', 'fallen', 'panic', 'intruder'
            'fall_frame': 110 if i == 5 else -1, # Agent 5 falls at ~4.4 sec
            'intruder': (i == 8),                # Agent 8 walks into track zone at ~9 sec
            'panic_speed': random.uniform(4.0, 7.0)
        })
        
    for frame_idx in range(total_frames):
        # Base background: Railway Station Platform
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Concourse floor (warm grey)
        frame[:, 0:720] = (160, 160, 165)
        # Platform Edge floor (slightly darker grey with textured tiles)
        frame[:, 720:1100] = (140, 140, 145)
        # Platform yellow safety line
        cv2.line(frame, (1080, 0), (1080, height), (0, 215, 255), 14)
        # Platform edge granite border
        cv2.line(frame, (1100, 0), (1100, height), (70, 70, 75), 12)
        
        # Track rail bed (dark ballast stone texture)
        frame[:, 1100:width] = (50, 50, 55)
        # Steel rails
        cv2.line(frame, (1160, 0), (1160, height), (170, 175, 180), 8)
        cv2.line(frame, (1240, 0), (1240, height), (170, 175, 180), 8)
        # Wooden ties / sleepers
        for y_tie in range(0, height, 40):
            cv2.line(frame, (1130, y_tie), (1270, y_tie), (35, 45, 60), 6)
            
        # Draw platform pillars & benches for realism
        cv2.rectangle(frame, (350, 300), (450, 340), (90, 90, 95), -1)
        cv2.putText(frame, "BENCH 1", (360, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        
        # Add station digital clock / timestamp watermark
        sec_elapsed = frame_idx / fps
        time_str = f"CAM-P1-CENTRAL | CCTV LIVE | 2026-09-22 18:30:{int(sec_elapsed):02d}.{int((sec_elapsed % 1)*100):02d}"
        cv2.putText(frame, time_str, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Anomaly simulation triggers based on time
        # Event 1: Fall around frame 110 (4.4s)
        # Event 2: Panic movement from frame 200 to 260 (8.0s - 10.4s)
        # Event 3: Intruder walks across yellow line to track zone from frame 270 (10.8s)
        
        is_panic_period = (200 <= frame_idx <= 260)
        
        # Update and render agents
        # Sort agents by Y coordinate for correct occlusion rendering (back to front)
        agents_sorted = sorted(agents, key=lambda a: a['y'])
        
        for agent in agents_sorted:
            # Fall logic for agent 5
            if agent['id'] == 5 and frame_idx >= agent['fall_frame']:
                if frame_idx < agent['fall_frame'] + 15:
                    # Falling animation: height rapidly decreases, width widens, y drops
                    agent['h'] = max(24, int(agent['h'] * 0.88))
                    agent['w'] = min(85, int(agent['w'] * 1.12))
                    agent['y'] += 3.5 # drops downwards
                    agent['vx'] *= 0.6
                    agent['vy'] *= 0.6
                else:
                    # Fully fallen on ground, stationary
                    agent['state'] = 'fallen'
                    agent['h'] = 24
                    agent['w'] = 82
                    agent['vx'] = 0.0
                    agent['vy'] = 0.0
            
            # Intruder logic for agent 8
            elif agent['intruder'] and frame_idx >= 150:
                # Walk rightwards toward rail bed
                agent['vx'] = 2.4
                agent['vy'] = 0.2
            
            # Panic period: sudden acceleration and chaotic scattering
            elif is_panic_period and agent['id'] != 5:
                # Disperse rapidly in random directions
                if frame_idx == 200:
                    angle = random.uniform(0, 2 * math.pi)
                    agent['vx'] = agent['panic_speed'] * math.cos(angle)
                    agent['vy'] = agent['panic_speed'] * math.sin(angle)
            elif frame_idx > 260 and agent['id'] != 5 and not agent['intruder']:
                # Slow back down after panic
                agent['vx'] *= 0.96
                agent['vy'] *= 0.96
                if abs(agent['vx']) < 0.8:
                    agent['vx'] = random.uniform(-1.0, 1.0)
                if abs(agent['vy']) < 0.5:
                    agent['vy'] = random.uniform(-0.5, 0.5)
            
            # Move agent
            agent['x'] += agent['vx']
            agent['y'] += agent['vy']
            
            # Bounce off borders (unless intruder entering track zone)
            max_x = 1260 if agent['intruder'] and frame_idx >= 150 else 1070
            if agent['x'] < 70:
                agent['x'] = 70
                agent['vx'] *= -1
            elif agent['x'] > max_x:
                agent['x'] = max_x
                agent['vx'] *= -1
                
            if agent['y'] < 160:
                agent['y'] = 160
                agent['vy'] *= -1
            elif agent['y'] > 670:
                agent['y'] = 670
                agent['vy'] *= -1
                
            # Render realistic person silhouette
            ax = int(agent['x'])
            ay = int(agent['y'])
            aw = int(agent['w'])
            ah = int(agent['h'])
            
            x1 = ax - aw // 2
            x2 = ax + aw // 2
            y1 = ay - ah // 2
            y2 = ay + ah // 2
            
            if agent['state'] == 'fallen':
                # Fallen person lying horizontally
                # Body/torso horizontal rectangle
                cv2.rectangle(frame, (x1, y1 + 5), (x2, y2), agent['shirt_color'], -1)
                # Head at one end
                cv2.circle(frame, (x1 + 8, ay), 10, agent['skin_color'], -1)
                # Legs at other end
                cv2.rectangle(frame, (x2 - 25, y1 + 6), (x2, y2 - 2), agent['pants_color'], -1)
            else:
                # Standing person
                # Head (circle at top)
                head_r = max(6, int(aw * 0.28))
                head_cy = y1 + head_r + 2
                cv2.circle(frame, (ax, head_cy), head_r, agent['skin_color'], -1)
                
                # Torso / shirt
                torso_y1 = head_cy + head_r
                torso_y2 = torso_y1 + int(ah * 0.45)
                cv2.rectangle(frame, (x1 + 3, torso_y1), (x2 - 3, torso_y2), agent['shirt_color'], -1)
                
                # Legs / pants
                legs_y1 = torso_y2
                legs_y2 = y2
                leg_w = max(4, (aw - 8) // 2)
                cv2.rectangle(frame, (x1 + 4, legs_y1), (x1 + 4 + leg_w, legs_y2), agent['pants_color'], -1)
                cv2.rectangle(frame, (x2 - 4 - leg_w, legs_y1), (x2 - 4, legs_y2), agent['pants_color'], -1)
                
                # Arms
                cv2.line(frame, (x1 + 2, torso_y1 + 4), (x1 + 1, torso_y2 - 2), agent['shirt_color'], 4)
                cv2.line(frame, (x2 - 2, torso_y1 + 4), (x2 - 1, torso_y2 - 2), agent['shirt_color'], 4)

        # Write frame to video
        out.write(frame)
        
    out.release()
    print(f"[generate_test_video] Successfully created CCTV test video: {output_path} ({total_frames} frames, {width}x{height})")

if __name__ == "__main__":
    create_synthetic_cctv_video()
