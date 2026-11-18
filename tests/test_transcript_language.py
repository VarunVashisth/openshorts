import copy
import pytest

import transcript_language as tl


def test_hinglish_forces_hindi_unless_urdu_explicit():
    assert tl.transcription_env('auto', 'hinglish')['WHISPER_LANGUAGE'] == 'hi'
    assert tl.transcription_env('ur', 'hinglish')['WHISPER_LANGUAGE'] == 'ur'
    with pytest.raises(ValueError):
        tl.transcription_env('en', 'hinglish')
    with pytest.raises(ValueError):
        tl.transcription_env('hi', 'unsupported')


def test_old_auto_detected_transcript_not_reused_for_forced_hindi(monkeypatch):
    monkeypatch.setenv('WHISPER_LANGUAGE', 'hi')
    assert not tl.language_matches({'language': 'hi'})
    assert tl.language_matches({'language': 'hi', 'requested_language': 'hi'})
    monkeypatch.setenv('WHISPER_LANGUAGE', 'auto')
    assert tl.language_matches({'language': 'ur'})


def test_romanization_preserves_timestamps_and_original(monkeypatch):
    monkeypatch.setenv('SUBTITLE_SCRIPT', 'hinglish')
    monkeypatch.setattr(tl.llm_backend, 'active', lambda: True)
    monkeypatch.setattr(tl.llm_backend, 'generate_json', lambda *a: (
        {'words': [{'id': 0, 'text': 'Kya'}, {'id': 1, 'text': 'hai?'}]}, None))
    transcript = {'language': 'hi', 'text': 'क्या scene है?', 'segments': [{
        'start': 2, 'end': 4, 'text': 'क्या scene है?', 'words': [
            {'word': ' क्या', 'start': 2, 'end': 2.5},
            {'word': ' scene', 'start': 2.5, 'end': 3},
            {'word': ' है?', 'start': 3, 'end': 4}]}]}
    original = copy.deepcopy(transcript)
    result = tl.romanize_hindi(transcript)
    assert result['text'] == 'Kya scene hai?'
    assert transcript == original
    assert result['segments'][0]['words'] == [
        {'word': ' Kya', 'start': 2, 'end': 2.5},
        {'word': ' scene', 'start': 2.5, 'end': 3},
        {'word': ' hai?', 'start': 3, 'end': 4}]
    assert tl.romanize_hindi(result) is result


def test_unrecoverable_output_fails_instead_of_corrupting_alignment(monkeypatch):
    monkeypatch.setenv('SUBTITLE_SCRIPT', 'hinglish')
    monkeypatch.setattr(tl.llm_backend, 'active', lambda: True)
    monkeypatch.setattr(tl.llm_backend, 'generate_json', lambda prompt, schema: (
        {'text': 'क्या'} if schema is tl.RomanizedText else {'words': [{'id': 2, 'text': 'kya'}]}, None))
    with pytest.raises(ValueError, match='could not romanize'):
        tl.romanize_hindi({'segments': [{'text': 'क्या', 'words': [
            {'word': 'क्या', 'start': 0, 'end': 1}]}]})


def test_forced_language_is_sent_to_whisper_and_explicit_language_wins(monkeypatch):
    import transcribe_backends as tb
    monkeypatch.setenv('WHISPER_LANGUAGE', 'hi')
    calls = []
    monkeypatch.setattr(tb, '_run_whisper_once', lambda path, **params: calls.append(params) or ([], None))
    tb.run_whisper_transcription('source.mp4')
    tb.run_whisper_transcription('source.mp4', language='en')
    assert calls == [{'language': 'hi'}, {'language': 'en'}]


def test_reordered_response_is_aligned_by_id(monkeypatch):
    monkeypatch.setattr(tl.llm_backend, 'generate_json', lambda *a: (
        {'words': [{'id': 1, 'text': 'hai'}, {'id': 0, 'text': 'kya'}]}, None))
    assert tl._romanize_batch([{'id': 0, 'text': 'क्या'}, {'id': 1, 'text': 'है'}]) == ['kya', 'hai']


def test_missing_and_duplicate_ids_recover_by_splitting(monkeypatch):
    import json
    calls = []
    def generate(prompt, schema):
        source = json.loads(prompt.split('Input:\n')[1])
        calls.append(len(source))
        if len(source) > 1:
            return {'words': [{'id': 0, 'text': 'wrong'}, {'id': 0, 'text': 'duplicate'}]}, None
        mapping = {'क्या': 'kya', 'है': 'hai', 'नहीं': 'nahi'}
        return {'words': [{'id': 0, 'text': mapping[source[0]['text']]}]}, None
    monkeypatch.setattr(tl.llm_backend, 'generate_json', generate)
    source = [{'id': i, 'text': t} for i, t in enumerate(['क्या', 'है', 'नहीं'])]
    assert tl._romanize_batch(source) == ['kya', 'hai', 'nahi']
    assert calls[0] == 3 and 1 in calls


def test_single_word_fallback_does_not_ask_model_for_ids(monkeypatch):
    def generate(prompt, schema):
        if schema is tl.RomanizedText:
            return {'text': 'kya'}, None
        return {'words': []}, None
    monkeypatch.setattr(tl.llm_backend, 'generate_json', generate)
    assert tl._romanize_batch([{'id': 0, 'text': 'क्या'}]) == ['kya']


def test_server_errors_do_not_trigger_recursive_retries(monkeypatch):
    calls = []
    def generate(*args):
        calls.append(1)
        raise RuntimeError('LLM server 503')
    monkeypatch.setattr(tl.llm_backend, 'generate_json', generate)
    with pytest.raises(RuntimeError, match='503'):
        tl._romanize_batch([{'id': 0, 'text': 'क्या'}])
    assert len(calls) == 1
