#!/usr/bin/env python3
"""Compare selected PH and Paired recovery rollouts with time-aligned detector plots."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WIDTH = 800
HEIGHT = 864
BACKGROUND = '#101827'
TEXT = '#e6edf7'
MUTED = '#a6b4c8'
COLORS = {'ph': '#22c1d6', 'paired': '#ba9bff'}
FONT_ROOT = Path('/usr/share/fonts/truetype/dejavu')


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_ROOT / f'DejaVuSans{"-Bold" if bold else ""}.ttf'), size)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def retreat_ranges(trace):
    steps = [event['step'] for event in trace if event['phase'] == 'retreat']
    ranges = []
    for step in steps:
        if ranges and step == ranges[-1][1] + 1:
            ranges[-1][1] = step
        else:
            ranges.append([step, step])
    return ranges


def load_run(folder, episode_id, label):
    folder = folder.resolve()
    run = json.loads((folder / 'run.json').read_text())
    rows = [json.loads(line) for line in (folder / 'episodes.jsonl').read_text().splitlines() if line]
    row = next(item for item in rows if item['episode_id'] == episode_id)
    trace = json.loads((folder / f'episode_{episode_id:03d}_trace.json').read_text())
    detections = [event for event in trace if event['phase'] == 'detection']
    threshold = run['threshold']
    if any(event['triggered'] != (event['anomaly_score'] > threshold) for event in detections):
        raise ValueError('Recorded detector decisions disagree with the frozen threshold')
    physical = [event for event in trace if event['phase'] in ('forward', 'retreat')]
    if [event['step'] for event in physical] != list(range(1, row['physical_steps'] + 1)):
        raise ValueError('Physical trace coverage is incomplete')
    outcome = 'success' if row['success'] else 'failed'
    video = folder / f'episode_{episode_id:03d}_{outcome}.mp4'
    original = folder / f'episode_{episode_id:03d}.mp4'
    if not video.exists() and original.exists():
        original.replace(video)
    reader = imageio.get_reader(video)
    frames = [Image.fromarray(frame) for frame in reader]
    metadata = reader.get_meta_data()
    reader.close()
    if len(frames) != row['physical_steps'] + 1 or abs(metadata['fps'] - 20) > .01:
        raise ValueError('Video must have the reset frame and exactly one frame per physical step at 20 fps')
    samples = [dict(step=0, score=row['initial_score'], triggered=False, kind='initial')]
    samples += [dict(step=event['step'], score=event['anomaly_score'],
                     triggered=event['triggered'], kind='policy_boundary') for event in detections]
    return dict(folder=folder, run=run, row=row, trace=trace, samples=samples, frames=frames,
                video=video, label=label, retreats=retreat_ranges(trace), physical=physical)


def verify_pair(left, right, expected_outcomes='ph-success-paired-failed'):
    keys = ('checkpoint_sha256', 'normalization_sha256', 'catalog_sha256', 'implementation_sha256',
            'checkpoint_step', 'training_seed', 'policy_dataset', 'policy_mode', 'action_conditioning_enabled',
            'scale', 'max_steps', 'negative_prompt_mode', 'protocol')
    for key in keys:
        if left['run'][key] != right['run'][key]:
            raise ValueError(f'Comparison changes more than detector: {key}')
    if (left['run']['policy_dataset'] != 'paired' or left['run']['negative_prompt_mode'] != 'full_plan'
            or left['row']['episode_id'] != right['row']['episode_id']
            or left['row']['environment_seed'] != right['row']['environment_seed']):
        raise ValueError('Require the same scene, Paired policy and full-plan prompt')
    if expected_outcomes == 'both-failed':
        if any(data['row']['success'] or data['row']['recoveries'] < 1 or not data['retreats']
               for data in (left, right)):
            raise ValueError('Require both rollouts to trigger physical retreat and fail')
    elif expected_outcomes == 'ph-success-paired-failed':
        if not left['row']['success'] or right['row']['success']:
            raise ValueError('Require PH success and Paired failure')
    else:
        raise ValueError(f'Unknown expected outcome: {expected_outcomes}')
    for data in (left, right):
        detector = Path(data['run']['detector'])
        calibration = json.loads((detector.parent / 'calibration.json').read_text())
        if (calibration['kappa'] != 4 or calibration['threshold'] != data['run']['threshold']
                or sha256(detector) != data['run']['detector_sha256']):
            raise ValueError('Frozen kappa=4 detector identity mismatch')


def configure_axis(axis, data, max_steps, ymax):
    axis.set_facecolor(BACKGROUND)
    axis.set_xlim(0, max_steps)
    axis.set_ylim(0, ymax)
    axis.set_xlabel('Physical control step (20 steps = 1 second)', color=MUTED, fontsize=10)
    axis.set_ylabel('AE reconstruction MSE', color=MUTED, fontsize=10)
    axis.tick_params(colors=MUTED, labelsize=9)
    for spine in axis.spines.values():
        spine.set_color('#43516a')
    axis.grid(color='#344257', alpha=.55, linewidth=.6)
    axis.axhline(data['run']['threshold'], color='#ffbd69', linestyle='--', linewidth=1.3)
    for start, end in data['retreats']:
        axis.axvspan(start - 1, end, color='#eeb760', alpha=.12)


def chart_background(data, max_steps, ymax):
    figure = Figure(figsize=(8, 3), dpi=100, facecolor=BACKGROUND)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes([.12, .21, .84, .64])
    configure_axis(axis, data, max_steps, ymax)
    axis.set_title(f"Error detector | threshold = {data['run']['threshold']:.6f}",
                   color=TEXT, fontsize=12, loc='left', pad=12)
    canvas.draw()
    image = Image.fromarray(np.asarray(canvas.buffer_rgba())[..., :3].copy())
    return image, axis.transData, canvas.get_width_height()[1]


def panel(data, step, max_steps, chart):
    effective = min(step, data['row']['physical_steps'])
    held = step > effective
    ended = effective == data['row']['physical_steps']
    image = Image.new('RGB', (WIDTH, HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    label = 'PH detector' if data['label'] == 'ph' else 'Paired detector'
    outcome = 'SUCCESS' if data['row']['success'] else 'FAILED'
    color = '#58df9a' if data['row']['success'] else '#ff7a88'
    draw.text((24, 14), label, fill=TEXT, font=font(28, True))
    draw.text((584, 18), outcome, fill=color, font=font(25, True))
    draw.text((24, 54), f"Can (Paired) | conditional 2k | S={data['run']['scale']:g} | kappa=4 | "
              f"episode {data['row']['episode_id']:03d}",
              fill=MUTED, font=font(17))
    phase = 'RESET' if effective == 0 else data['physical'][effective - 1]['phase'].upper()
    if ended:
        phase = 'SUCCESS - HOLD' if data['row']['success'] else \
                ('FAILED - ROLLBACK' if data['row']['termination'] == 'rollback_failed' else 'FAILED - BUDGET')
    triggers = sum(sample['triggered'] for sample in data['samples'] if sample['step'] <= effective)
    draw.text((24, 82), f't={step / 20:5.2f}s | step {effective:03d}/{max_steps} | {phase} | triggers {triggers}',
              fill=color if ended else TEXT, font=font(17))
    background, transform, chart_height = chart
    image.paste(background, (0, 105))
    draw = ImageDraw.Draw(image)

    def point(sample_step, score):
        position = transform.transform((sample_step, score))
        return float(position[0]), float(105 + chart_height - position[1])

    samples = [sample for sample in data['samples'] if sample['step'] <= effective]
    points = [point(sample['step'], sample['score']) for sample in samples]
    if len(points) > 1:
        draw.line(points, fill=COLORS[data['label']], width=3)
    for sample, (horizontal, vertical) in zip(samples, points):
        radius = 5 if sample['triggered'] else 3
        draw.ellipse((horizontal-radius, vertical-radius, horizontal+radius, vertical+radius),
                     fill='#ff7182' if sample['triggered'] else COLORS[data['label']])
    horizontal = point(effective, 0)[0]
    plot_top = 105 + chart_height * .15
    draw.line((horizontal, plot_top, horizontal, point(effective, 0)[1]), fill='#f0f4fc', width=1)
    last = samples[-1]
    draw.text((24, 405), f"Last measured MSE: {last['score']:.6f} at step {last['step']} | orange: retreat",
              fill=MUTED, font=font(16))
    draw.text((24, 430), 'Agent view', fill=TEXT, font=font(16, True))
    draw.text((416, 430), 'Wrist view', fill=TEXT, font=font(16, True))
    frame = data['frames'][effective]
    middle = frame.width // 2
    for horizontal, crop in [(12, (0, 0, middle, frame.height)), (404, (middle, 0, frame.width, frame.height))]:
        view = frame.crop(crop).resize((384, 384), Image.Resampling.LANCZOS)
        image.paste(view, (horizontal, 454))
    footer = 'Final frame held; shared physical-time clock continues.' if held else \
             'Recorded detector samples; connecting lines do not add measurements.'
    draw.text((18, 842), footer, fill=MUTED, font=font(14))
    return np.asarray(image)


def write_static_curves(output, runs, max_steps, ymax):
    figure = Figure(figsize=(16, 4.6), dpi=150, facecolor=BACKGROUND)
    axes = figure.subplots(1, 2)
    for axis, data in zip(axes, runs):
        configure_axis(axis, data, max_steps, ymax)
        samples = data['samples']
        axis.plot([sample['step'] for sample in samples], [sample['score'] for sample in samples],
                  color=COLORS[data['label']], marker='o', markersize=3)
        triggered = [sample for sample in samples if sample['triggered']]
        axis.scatter([sample['step'] for sample in triggered], [sample['score'] for sample in triggered],
                     color='#ff7182', s=36, zorder=5)
        axis.set_title(f"{data['label'].upper()} detector | {'SUCCESS' if data['row']['success'] else 'FAILED'} | "
                       f"threshold {data['run']['threshold']:.6f}", color=TEXT, fontsize=12)
        axis.axvline(data['row']['physical_steps'], color=TEXT, linestyle=':', linewidth=1)
    figure.legend(handles=[Line2D([], [], color='#ffbd69', linestyle='--', label='Frozen threshold'),
                           Line2D([], [], color='#ff7182', marker='o', linestyle='', label='Recovery trigger')],
                  loc='upper center', ncols=2, facecolor=BACKGROUND, edgecolor=BACKGROUND, labelcolor=TEXT)
    figure.subplots_adjust(top=.82, bottom=.16, left=.06, right=.98, wspace=.21)
    figure.savefig(output / 'detector_curves.png', facecolor=BACKGROUND)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ph-run', type=Path, required=True)
    parser.add_argument('--paired-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode-id', type=int, default=27)
    parser.add_argument('--expected-outcomes', choices=('ph-success-paired-failed', 'both-failed'),
                        default='ph-success-paired-failed')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    runs = [load_run(args.ph_run, args.episode_id, 'ph'), load_run(args.paired_run, args.episode_id, 'paired')]
    verify_pair(*runs, args.expected_outcomes)
    max_steps = runs[0]['run']['max_steps']
    ymax = max(max(sample['score'] for sample in data['samples']) for data in runs) * 1.18
    ymax = max(ymax, max(data['run']['threshold'] for data in runs) * 1.15)
    charts = [chart_background(data, max_steps, ymax) for data in runs]
    total_steps = max(data['row']['physical_steps'] for data in runs)
    outcomes = ['success' if data['row']['success'] else 'failed' for data in runs]
    prefix = f'episode_{args.episode_id:03d}'
    combined = output / f'{prefix}_ph_{outcomes[0]}_vs_paired_{outcomes[1]}_side_by_side.mp4'
    single_paths = [output / f'{prefix}_{data["label"]}_detector_{outcome}.mp4'
                    for data, outcome in zip(runs, outcomes)]
    if any(path.exists() for path in [combined, *single_paths]):
        raise FileExistsError('Rendered output already exists; choose a fresh output directory')
    writers = []
    try:
        for path in [*single_paths, combined]:
            writers.append(imageio.get_writer(path, fps=20, codec='libx264', macro_block_size=1,
                                              ffmpeg_params=['-crf', '18', '-preset', 'fast', '-movflags', '+faststart']))
        for step in range(total_steps + 1):
            panels = [panel(data, step, max_steps, chart) for data, chart in zip(runs, charts)]
            for writer, data, image in zip(writers[:2], runs, panels):
                if step <= data['row']['physical_steps']:
                    writer.append_data(image)
            frame = np.concatenate(panels, axis=1)
            writers[2].append_data(frame)
            if step in (80, runs[0]['row']['physical_steps'], total_steps) or any(
                    sample['step'] == step and sample['triggered'] for data in runs for sample in data['samples']):
                Image.fromarray(frame).save(output / f'comparison_step_{step:03d}.png')
            if step % 100 == 0:
                print(f'Encoded physical step {step}/{total_steps}', flush=True)
    finally:
        for writer in writers:
            writer.close()
    write_static_curves(output, runs, max_steps, ymax)
    with (output / 'detector_samples.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['detector', 'step', 'time_seconds', 'score', 'threshold', 'triggered', 'kind'])
        writer.writeheader()
        for data in runs:
            for sample in data['samples']:
                writer.writerow(dict(detector=data['label'], time_seconds=sample['step']/20,
                                     threshold=data['run']['threshold'], **sample))
    manifest = dict(episode_id=args.episode_id, expected_outcomes=args.expected_outcomes,
                    fps=20, synchronization='physical_step_including_retreat',
                    shorter_video='freeze_final_frame', plot='Recorded initial and policy-boundary AE MSE only; no detector recomputation',
                    native_camera_resolution=[84,84], camera_display_resolution=[384,384],
                    comparison_video=str(combined), individual_videos=list(map(str,single_paths)),
                    comparison_frames=total_steps+1, comparison_resolution=[1600,864], runs=[])
    for data in runs:
        manifest['runs'].append(dict(label=data['label'], source=str(data['folder']),
            result=data['row'], run=data['run'], retreat_ranges=data['retreats'], raw_video=str(data['video']),
            trace_sha256=sha256(data['folder']/f'episode_{args.episode_id:03d}_trace.json')))
    (output/'visualization_manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'README.md').write_text(
        '# Can(Paired) detector 对照视频\n\n'
        f"固定 Paired conditional 2k、seed 0、S={runs[0]['run']['scale']:g}、完整 action_pred negative prompt、κ=4、"
        f"episode {args.episode_id}（环境 seed {runs[0]['row']['environment_seed']}），仅更换 PH / Paired detector。两侧以当前同一代码重跑，"
        'checkpoint、归一化、catalog、implementation、物理预算与协议均核对一致。\n\n'
        f'主文件：`{combined.name}`。'
        f'左侧 PH {outcomes[0]}，右侧 Paired {outcomes[1]}；各自上方是实际日志中的 AE reconstruction MSE 曲线。'
        '橙色虚线是各自冻结阈值，红点为触发回退，橙色背景区间为物理回退，白色竖线标记当前步。'
        '两图使用相同横纵轴；曲线只使用初始分数和实际策略调用边界的检测分数，连线不表示额外测量。\n\n'
        '20 fps 对齐物理控制时间（包含回退），不按各自完成百分比拉伸。较早结束的一侧保持最后一帧，'
        '直到另一侧结束。两机位原生分辨率均为84×84，仅放大显示，不产生额外细节。\n\n'
        f'两个单独带曲线视频分别为 `{single_paths[0].name}` 和 '
        f'`{single_paths[1].name}`；原始无标注双机位视频在各 detector 目录中。'
        '完整曲线静态图为 `detector_curves.png`，原始分数为 `detector_samples.csv`，'
        '来源与身份核对见 `visualization_manifest.json`。\n\n'
        f'本例是从既有评测中按“{args.expected_outcomes}”挑选的说明性场景；'
        '两局重跑不作为新增50局基准评测，单例也不能证明detector的一般优劣。\n')
    print(f'Completed: {combined}', flush=True)


if __name__ == '__main__':
    main()
