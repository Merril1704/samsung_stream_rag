import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from controller import SessionState, RuleBasedController, ModelBasedController, MockLLMClient, HybridController

SCENARIO_1 = [
    (0.0, "I need to plan a customer workshop in"),
    (0.8, "Pune for 30 people, and I need"),
    (1.6, "the cancellation policy and the catering options."),
]

SCENARIO_3 = [
    (0.0, "Please repeat your last answer in two bullet points."),
]


def run_scenario(controller, chunks, session_id, last_answer_topic=None):
    state = SessionState(session_id=session_id)
    state.last_answer_topic = last_answer_topic
    print(f"\n--- {controller.mode} :: {session_id} ---")
    for ts, chunk in chunks:
        d = controller.decide(state, chunk, ts)
        print(f"[{ts:>4.1f}s] chunk={chunk!r:52} -> {d.action:9} "
              f"trigger={d.trigger:12} conf={d.confidence:.2f}  {d.reason}")


if __name__ == "__main__":
    controllers = [
        RuleBasedController(wait_threshold=4),
        ModelBasedController(MockLLMClient()),
        HybridController(RuleBasedController(wait_threshold=4), ModelBasedController(MockLLMClient())),
    ]

    print("=" * 70)
    print("SCENARIO 1 — Incremental Multi-Intent Utterance")
    print("=" * 70)
    for ctrl in controllers:
        run_scenario(ctrl, SCENARIO_1, "scenario1")

    print("\n" + "=" * 70)
    print("SCENARIO 3 — Query Suppression")
    print("=" * 70)
    for ctrl in controllers:
        run_scenario(ctrl, SCENARIO_3, "scenario3",
                     last_answer_topic="venue capacity cancellation catering Pune workshop")
