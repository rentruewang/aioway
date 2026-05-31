# Copyright (c) AIoWay Authors - All Rights Reserved

import pytest

from aioway._sess import Session


class SubSession(Session):
    pass


@pytest.fixture
def sess_1():
    yield Session()


@pytest.fixture
def sess_2():
    yield Session()


def test_session_scope(sess_1: Session, sess_2: Session):
    assert not sess_1.is_active
    assert not sess_2.is_active

    with sess_1():
        assert sess_1.is_active
        assert not sess_2.is_active

        with sess_2():
            assert sess_2.is_active


def test_current_session(sess_1: Session, sess_2: Session):
    with sess_1():
        assert Session.current() is sess_1

        with sess_2():
            assert Session.current() is sess_2

        assert Session.current() is sess_1


def test_no_repeat_entry(sess_1: Session):
    with sess_1():
        assert Session.current() is sess_1

        with pytest.raises(RuntimeError):
            with sess_1():
                pass
