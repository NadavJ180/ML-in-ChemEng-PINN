# Restarts of the confirmation training (#40)

**No restarts.** No run crashed or was stopped in the middle of a case, so each of the 30 cases was trained exactly once, from its own seeds.

One thing to know: before the fix in commit `ab71414` (PR #52, 23 Oct 2026), each run stopped with a `KeyError` right after its first case, while drawing an optional demo figure. By then that case was fully saved, so nothing was lost or trained again; the next run simply went on with the next case. This happened in the two runs that trained `case_32` and `case_37`. Those two cases were trained at commit `e7150d8`, the other 28 at `2442c8d`. The training code is the same in both: the fix only touched the demo figure.
