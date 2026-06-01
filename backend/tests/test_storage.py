from concurrent.futures import ThreadPoolExecutor

from speech_api.services.storage import read_json, write_json_atomic


def test_atomic_json_writer_uses_independent_temp_files_for_parallel_writes(tmp_path) -> None:
    state_path = tmp_path / "state.json"

    def write_value(index: int) -> None:
        write_json_atomic(state_path, {"index": index})

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(write_value, range(40)))

    assert isinstance(read_json(state_path, {}), dict)
