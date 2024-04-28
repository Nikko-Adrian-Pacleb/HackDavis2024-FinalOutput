"use client";

import { useEffect, useState } from "react";

import Image from "next/image";
import Link from "next/link";
export default function Home() {
  return (
    <main className="flex h-screen  bg-slate-100 flex-col items-center justify-between p-40">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm bg-slate-100 ">
        <h1 className="text-8xl px-24 py-16 text-center text-blue-300">
          {" "}
          <b>
            Welcome to <a className="text-blue-400">Sentimatic</a>
          </b>
        </h1>
        <div className="text-center">
          <Link
            className="text-slate-100 bg-blue-300 py-4 px-24  space-x-2 text-xl inline hover:bg-blue-200 "
            href="/patients"
          >
            Click to load images
          </Link>
        </div>
      </div>
    </main>
  );
}
