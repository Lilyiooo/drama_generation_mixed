import cos_orm


class BaseCosModel(cos_orm.BaseCosModel):

    def marshal(self) -> bytes:
        if hasattr(self, "to_json"):
            rsp = self.to_json(ensure_ascii=False, indent=2)
            if isinstance(rsp, str):
                rsp = rsp.encode("utf-8")
            return rsp
        else:
            raise NotImplementedError()
