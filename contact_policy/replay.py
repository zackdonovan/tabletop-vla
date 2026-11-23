"""Run a labeled closed-loop demonstration on a new deterministic preview scene."""
from pathlib import Path
import argparse
import json

from PIL import Image

from env import WORDS
from sim import Physical, config
from learn import Learner
from eval import episode

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--puck", choices=WORDS[:3], default="red")
    parser.add_argument("--pad", choices=WORDS[3:], default="yellow")
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--condition", choices=("nominal", "lowfric", "offset"), default="nominal")
    parser.add_argument("--gif", type=Path, default=Path("preview/grasp.gif"))
    args = parser.parse_args()
    if args.scene < 0:
        parser.error("--scene must be nonnegative")
    goal = (WORDS.index(args.puck), WORDS.index(args.pad)-3)
    robot = Physical(render=True)
    try:
        frames = []
        result = episode(robot, Learner(), config("vla-preview-v1", args.scene), args.scene,
                         goal, goal, args.condition, "full", 2, frames)
        frames.append(robot.rgb())
    finally:
        if robot.renderer is not None:
            robot.renderer.close()
    # Retain the actual last state after settling, even if the stride would omit it.
    sampled = frames[:-1:2] + [frames[-1]]
    images = [Image.fromarray(x).resize((384,384), Image.Resampling.NEAREST) for x in sampled]
    args.gif.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.gif, save_all=True, append_images=images[1:] + [images[-1]]*7,
                   loop=0, duration=100)
    print(json.dumps(dict(scene=args.scene, condition=args.condition, puck=args.puck,
                          pad=args.pad, gif=str(args.gif),
                          **{k:result[k] for k in ('success','checks','steps','lift_m','error_m')}), indent=2))

if __name__ == "__main__":
    main()
