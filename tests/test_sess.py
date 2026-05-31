# Copyright (c) AIoWay Authors - All Rights Reserved

import contextlib as ctxl

import pytest

from aioway.sess import Session


class SubSession(Session["SubSession"]):
    @ctxl.contextmanager
    def do(self):
        yield self


@pytest.fixture
def sess_1():
    yield SubSession()


@pytest.fixture
def sess_2():
    yield SubSession()


def test_session_scope(sess_1: SubSession, sess_2: SubSession):
    assert not sess_1.is_active
    assert not sess_2.is_active

    with sess_1():
        assert sess_1.is_active
        assert not sess_2.is_active

        with sess_2():
            assert sess_2.is_active


def test_current_session(sess_1: SubSession, sess_2: SubSession):
    with sess_1():
        assert SubSession.current() is sess_1

        with sess_2():
            assert SubSession.current() is sess_2

        assert SubSession.current() is sess_1


def test_no_repeat_entry(sess_1: Session):
    with sess_1():
        assert SubSession.current() is sess_1

        with pytest.raises(RuntimeError):
            with sess_1():
                pass


def test_no_abstract_session_current():
    with pytest.raises(RuntimeError):
        Session.current()
